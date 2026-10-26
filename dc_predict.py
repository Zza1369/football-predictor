import io
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

warnings.filterwarnings("ignore")

LOCAL_CSV = "fd_with_xg_2526.csv"
SEASON_2627 = "2627"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# 五大联赛+英冠
LEAGUE_NAMES = {
    "E0": "英超", "SP1": "西甲", "D1": "德甲",
    "I1": "意甲", "F1": "法甲", "E1": "英冠"
}

# 【核心开关】设为 True：自动抓取未来几天的所有比赛；设为 False：只跑 fixtures.csv
AUTO_MODE = True
DAYS_AHEAD = 3  # 预测未来几天内的比赛

MAX_GOALS = 8
TOP_N = 6

# 加载中文字典
def load_dict():
    try:
        df = pd.read_csv("teams_cn.csv", encoding="utf-8-sig")
        return dict(zip(df["en"].str.strip(), df["cn"].str.strip()))
    except Exception:
        return {}

EN_TO_CN = load_dict()

def translate_to_cn(name):
    """如果是英文字典里有的，转成中文；没有就保留英文"""
    return EN_TO_CN.get(str(name).strip(), str(name).strip())

def _std(df, keep_unplayed=False):
    df = df.copy()
    df = df.rename(columns={"HxG": "xg_home", "AxG": "xg_away"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    
    # 把没有比分的比赛标为 NaN
    df["hg"] = pd.to_numeric(df.get("hg"), errors="coerce")
    df["ag"] = pd.to_numeric(df.get("ag"), errors="coerce")
    
    if not keep_unplayed:
        # 拟合模型时，只保留已经踢完的比赛
        df = df.dropna(subset=["hg", "ag"])
        
    df["xg_home"] = pd.to_numeric(df.get("xg_home"), errors="coerce")
    df["xg_away"] = pd.to_numeric(df.get("xg_away"), errors="coerce")
    return df[["date", "home", "away", "hg", "ag", "xg_home", "xg_away"]]

def load_league(code):
    """加载联赛数据，同时保留已完赛（训练）和未完赛（预测）的数据"""
    frames_train = []
    frames_future = []
    
    # 2526 历史数据（只用于训练）
    try:
        local = pd.read_csv(LOCAL_CSV, encoding="utf-8-sig")
        local = local[local["league"] == LEAGUE_NAMES[code]].copy()
        if len(local):
            frames_train.append(_std(local, keep_unplayed=False))
    except FileNotFoundError:
        pass

    # 2627 最新数据（区分训练和预测）
    url = BASE_URL.format(season=SEASON_2627, league=code)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(io.StringIO(text))
        raw = raw.rename(columns={
            "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
            "FTHG": "hg", "FTAG": "ag",
        })
        # 训练数据：有比分
        frames_train.append(_std(raw, keep_unplayed=False))
        # 预测数据：保留没比分的未来赛程
        frames_future.append(_std(raw, keep_unplayed=True))
    except Exception as e:
        print(f"    {code} 2627 数据拉取失败: {type(e).__name__}")

    if not frames_train:
        raise RuntimeError(f"{code} 无训练数据")
        
    df_train = pd.concat(frames_train, ignore_index=True)
    df_train = df_train.drop_duplicates(subset=["date", "home", "away"]).sort_values("date")
    
    df_future = pd.DataFrame()
    if frames_future:
        df_future = pd.concat(frames_future, ignore_index=True)
        df_future = df_future.drop_duplicates(subset=["date", "home", "away"]).sort_values("date")
        # 过滤出：没比分 + 日期在 [今天-1天, 今天+N天] 之间
        today = pd.Timestamp.now().normalize()
        end = today + pd.Timedelta(days=DAYS_AHEAD)
        df_future = df_future[
            (df_future["date"] >= today - pd.Timedelta(days=1)) & 
            (df_future["date"] <= end) & 
            (df_future["hg"].isna())
        ]

    return df_train.reset_index(drop=True), df_future.reset_index(drop=True)

# 泊松相关
def poisson_deviance(y, mu):
    term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (term - (y - mu))

def rho_correction(hg, ag, lh, la, rho):
    if hg == 0 and ag == 0: return 1 - lh * la * rho
    if hg == 0 and ag == 1: return 1 + lh * rho
    if hg == 1 and ag == 0: return 1 + la * rho
    if hg == 1 and ag == 1: return 1 - rho
    return 1.0

class DixonColes:
    def __init__(self, xi=0.0019, use_xg=True):
        self.xi = xi
        self.use_xg = use_xg

    def _unpack(self, p, n):
        return p[:n], p[n:2*n], p[2*n]

    def _loss(self, p, df, n, w):
        a, d, h = self._unpack(p, n)
        hi, ai = df["home_idx"].values, df["away_idx"].values
        lh = np.exp(a[hi] - d[ai] + h)
        la = np.exp(a[ai] - d[hi])
        if self.use_xg:
            loss = poisson_deviance(df["xg_home"].values, lh) + \
                   poisson_deviance(df["xg_away"].values, la)
        else:
            loss = -(poisson.logpmf(df["hg"].values, lh) +
                     poisson.logpmf(df["ag"].values, la))
        return np.sum(w * loss)

    def fit(self, df):
        df = df.copy()
        teams = sorted(set(df["home"]) | set(df["away"]))
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        df["home_idx"] = df["home"].map(self._idx)
        df["away_idx"] = df["away"].map(self._idx)
        days = (df["date"].max() - df["date"]).dt.days.values
        w = np.exp(-self.xi * days)

        if self.use_xg:
            mask = (df["xg_home"].notna() & df["xg_away"].notna()).values
            cov = mask.sum() / len(df) if len(df) else 0
            if cov < 0.5:
                self.use_xg = False
            else:
                df_fit = df[mask].reset_index(drop=True)
                w_fit = w[mask]
        if not self.use_xg:
            df_fit = df.reset_index(drop=True)
            w_fit = w

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1]])
        res = minimize(self._loss, x0, args=(df_fit, n, w_fit),
                       method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
        self.attack, self.defence, self.home_adv = self._unpack(res.x, n)
        self._fit_rho(df, w)
        return self

    def _fit_rho(self, df, w):
        hi, ai = df["home_idx"].values, df["away_idx"].values
        hg, ag = df["hg"].values, df["ag"].values
        lh = np.exp(self.attack[hi] - self.defence[ai] + self.home_adv)
        la = np.exp(self.attack[ai] - self.defence[hi])

        def nll(rho):
            t = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            t[m00] = 1 - lh[m00] * la[m00] * rho
            t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho
            t[m11] = 1 - rho
            t = np.clip(t, 1e-10, None)
            ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la) + np.log(t)
            return -np.sum(w * ll)

        r = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(r.x)

    def predict(self, home, away, max_goals=MAX_GOALS):
        hi, ai = self._idx[home], self._idx[away]
        lh = np.exp(self.attack[hi] - self.defence[ai] + self.home_adv)
        la = np.exp(self.attack[ai] - self.defence[hi])
        g = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lh) * poisson.pmf(a, la)
                p *= rho_correction(h, a, lh, la, self.rho)
                g[h, a] = p
        g = np.clip(g, 0, None)
        g /= g.sum()
        return g, lh, la

def main():
    print("=" * 64)
    print(f"  全自动预测系统 | {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 64)

    results = []
    
    for code, name in LEAGUE_NAMES.items():
        print(f"\n[{name}] 加载数据并拟合 ...")
        try:
            df_train, df_future = load_league(code)
            model = DixonColes(use_xg=True).fit(df_train)
        except Exception as e:
            print(f"  处理失败: {e}")
            continue

        # 确定要预测的目标
        targets = []
        if AUTO_MODE:
            if not df_future.empty:
                targets = [(r["home"], r["away"]) for _, r in df_future.iterrows()]
                print(f"  自动抓取到未来 {DAYS_AHEAD} 天内 {len(targets)} 场未开赛比赛")
            else:
                print(f"  未来 {DAYS_AHEAD} 天内没有未开赛的比赛")
        else:
            # 如果关掉了 AUTO_MODE，就退回读 fixtures.csv（此逻辑省略，推荐全自动）
            pass

        # 进行预测
        for h_en, a_en in targets:
            if h_en not in model._idx or a_en not in model._idx:
                print(f"    跳过: {h_en} vs {a_en} (数据不足)")
                continue

            g, lh, la = model.predict(h_en, a_en)
            hw = float(np.tril(g, -1).sum())
            dr = float(np.trace(g))
            aw = float(np.triu(g, 1).sum())
            scores = [(f"{h}-{a}", float(g[h, a]))
                      for h in range(g.shape[0]) for a in range(g.shape[1])]
            scores.sort(key=lambda x: x[1], reverse=True)

            h_cn = translate_to_cn(h_en)
            a_cn = translate_to_cn(a_en)

            print(f"    -> {h_cn} vs {a_cn} | 主胜 {hw:.1%} 平 {dr:.1%} 客胜 {aw:.1%}")

            results.append({
                "league": name,
                "home": h_cn,  # 存中文
                "away": a_cn,  # 存中文
                "lambda_home": round(lh, 3),
                "lambda_away": round(la, 3),
                "home_win": round(hw, 4),
                "draw": round(dr, 4),
                "away_win": round(aw, 4),
                "top_scores": "; ".join(f"{s}:{p:.3f}" for s, p in scores[:5]),
            })

    today = datetime.now().strftime("%Y-%m-%d")
    if results:
        pd.DataFrame(results).to_csv(f"predictions_{today}.csv",
                                     index=False, encoding="utf-8-sig")
        with open(f"predictions_{today}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 成功预测 {len(results)} 场比赛，已写入 predictions_{today}.csv")
    else:
        print("\n今日没有可以预测的比赛。")
        pd.DataFrame([{"message": "no predictions"}]).to_csv(
            f"predictions_{today}.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
