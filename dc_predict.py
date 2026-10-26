import io
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

warnings.filterwarnings("ignore")

LOCAL_CSV = "fd_with_xg_2526.csv"
FIXTURES_CSV = "fixtures.csv"
DICT_CSV = "teams_cn.csv"
SEASON_2627 = "2627"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUE_NAMES = {
    "E0": "英超", "SP1": "西甲", "D1": "德甲",
    "I1": "意甲", "F1": "法甲", "E1": "英冠"
}

MAX_GOALS = 8
TOP_N = 6

def load_dict():
    try:
        df = pd.read_csv(DICT_CSV, encoding="utf-8-sig")
        return dict(zip(df["cn"].str.strip(), df["en"].str.strip()))
    except Exception:
        return {}

CN_DICT = load_dict()

def translate(name):
    name = str(name).strip()
    return CN_DICT.get(name, name)

def _std(df):
    df = df.copy()
    df = df.rename(columns={"HxG": "xg_home", "AxG": "xg_away"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "hg", "ag"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["xg_home"] = pd.to_numeric(df.get("xg_home"), errors="coerce")
    df["xg_away"] = pd.to_numeric(df.get("xg_away"), errors="coerce")
    return df[["date", "home", "away", "hg", "ag", "xg_home", "xg_away"]]

def load_league(code):
    frames = []
    try:
        local = pd.read_csv(LOCAL_CSV, encoding="utf-8-sig")
        local = local[local["league"] == LEAGUE_NAMES[code]].copy()
        if len(local):
            frames.append(_std(local))
    except FileNotFoundError:
        pass

    url = BASE_URL.format(season=SEASON_2627, league=code)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig", errors="replace")
        raw = pd.read_csv(io.StringIO(text))
        raw = raw.rename(columns={
            "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
            "FTHG": "hg", "FTAG": "ag",
        })
        frames.append(_std(raw))
    except Exception as e:
        print(f"    {code} 2627 跳过: {type(e).__name__}")

    if not frames:
        raise RuntimeError(f"{code} 无数据")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date")
    return df.reset_index(drop=True)

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

def load_fixtures():
    try:
        df = pd.read_csv(FIXTURES_CSV, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"  找不到 {FIXTURES_CSV}")
        return []
    df = df.dropna(subset=["league", "home", "away"])
    out = []
    for _, r in df.iterrows():
        code = str(r["league"]).strip()
        if code not in LEAGUE_NAMES:
            print(f"  跳过未知联赛代码: {code}")
            continue
        h_en = translate(r["home"])
        a_en = translate(r["away"])
        out.append((code, h_en, a_en, str(r["home"]).strip(), str(r["away"]).strip()))
    return out

def main():
    print(f"  全自动预测系统 | {datetime.now():%Y-%m-%d %H:%M}")
    targets = load_fixtures()
    print(f"从 fixtures.csv 读到 {len(targets)} 场比赛")

    if not targets:
        today = datetime.now().strftime("%Y-%m-%d")
        pd.DataFrame([{"message": "no fixtures"}]).to_csv(f"predictions_{today}.csv", index=False, encoding="utf-8-sig")
        return

    results = []
    cache = {}
    fitted = {}

    for code, h_en, a_en, h_cn, a_cn in targets:
        name = LEAGUE_NAMES[code]

        if code not in cache:
            print(f"[{name}] 加载数据 ...")
            try:
                cache[code] = load_league(code)
            except Exception as e:
                print(f"  加载失败: {e}, 跳过 {h_cn} vs {a_cn}")
                cache[code] = None
                continue

        if cache[code] is None:
            continue

        if code not in fitted:
            print(f"[{name}] 拟合 ...")
            try:
                fitted[code] = DixonColes(use_xg=True).fit(cache[code])
            except Exception as e:
                print(f"  拟合失败: {e}, 跳过")
                fitted[code] = None
                continue

        model = fitted[code]
        if model is None:
            continue

        if h_en not in model._idx or a_en not in model._idx:
            print(f"[{name}] {h_cn} vs {a_cn} (翻译为 {h_en} vs {a_en}): 队名不在数据中，跳过")
            print(f"  可用球队示例: {sorted(model._idx.keys())[:8]}")
            continue

        g, lh, la = model.predict(h_en, a_en)
        hw = float(np.tril(g, -1).sum())
        dr = float(np.trace(g))
        aw = float(np.triu(g, 1).sum())
        scores = [(f"{h}-{a}", float(g[h, a])) for h in range(g.shape[0]) for a in range(g.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"[{name}] {h_cn} vs {a_cn}  主胜 {hw:.1%} | 平局 {dr:.1%} | 客胜 {aw:.1%}")

        results.append({
            "league": name, "home": h_cn, "away": a_cn,
            "lambda_home": round(lh, 3), "lambda_away": round(la, 3),
            "home_win": round(hw, 4), "draw": round(dr, 4), "away_win": round(aw, 4),
            "top_scores": "; ".join(f"{s}:{p:.3f}" for s, p in scores[:5]),
        })

    today = datetime.now().strftime("%Y-%m-%d")
    if results:
        pd.DataFrame(results).to_csv(f"predictions_{today}.csv", index=False, encoding="utf-8-sig")
        with open(f"predictions_{today}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已写入 predictions_{today}.csv / .json")
    else:
        print("所有比赛均被跳过。")
        pd.DataFrame([{"message": "no predictions"}]).to_csv(f"predictions_{today}.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
