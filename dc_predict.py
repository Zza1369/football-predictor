import io
import os
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
SEASON_2627 = "2627"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

LEAGUE_NAMES = {
    "E0": "英超", "SP1": "西甲", "D1": "德甲",
    "I1": "意甲", "F1": "法甲", "E1": "英冠"
}

MAX_GOALS = 8
TOP_N = 6
MARKET_WEIGHT = 0.5

EN_TO_CN = {
    "Man United": "曼联", "Man City": "曼城", "Liverpool": "利物浦", "Arsenal": "阿森纳", "Chelsea": "切尔西",
    "Tottenham": "热刺", "Newcastle": "纽卡", "Aston Villa": "阿斯顿维拉", "Brighton": "布莱顿",
    "West Ham": "西汉姆联", "Crystal Palace": "水晶宫", "Wolves": "狼队", "Bournemouth": "伯恩茅斯",
    "Fulham": "富勒姆", "Everton": "埃弗顿", "Brentford": "布伦特福德", "Nott'm Forest": "诺丁汉森林",
    "Leicester": "莱斯特城", "Ipswich": "伊普斯维奇", "Southampton": "南安普顿",
    "Leeds": "利兹联", "Sunderland": "桑德兰", "Burnley": "伯恩利", "Sheffield United": "谢菲联",
    "Real Madrid": "皇马", "Barcelona": "巴萨", "Ath Madrid": "马竞", "Ath Bilbao": "毕尔巴鄂竞技",
    "Sociedad": "皇家社会", "Betis": "皇家贝蒂斯", "Villarreal": "比利亚雷亚尔", "Valencia": "瓦伦西亚",
    "Girona": "赫罗纳", "Celta": "塞尔塔", "Vallecano": "巴列卡诺", "Osasuna": "奥萨苏纳",
    "Mallorca": "马洛卡", "Getafe": "赫塔菲", "Alaves": "阿拉维斯", "Las Palmas": "拉斯帕尔马斯",
    "Leganes": "莱加内斯", "Valladolid": "巴拉多利德", "Espanol": "西班牙人", "Sevilla": "塞维利亚", "Oviedo": "奥维耶多",
    "Levante": "莱万特", "Bayern Munich": "拜仁", "Leverkusen": "勒沃库森", "Dortmund": "多特蒙德", 
    "RB Leipzig": "莱比锡红牛", "Eintracht Frankfurt": "法兰克福", "Stuttgart": "斯图加特", "Freiburg": "弗莱堡", 
    "Wolfsburg": "沃尔夫斯堡", "Mainz": "美因茨", "Hoffenheim": "霍芬海姆", "Augsburg": "奥格斯堡", 
    "Werder Bremen": "不莱梅", "Union Berlin": "柏林联合", "M'gladbach": "门兴", "St Pauli": "圣保利", 
    "Heidenheim": "海登海姆", "Holstein Kiel": "荷尔斯泰因基尔", "Bochum": "波鸿", "Hamburg": "汉堡", 
    "FC Koln": "科隆", "Inter": "国米", "Milan": "AC米兰", "Juventus": "尤文图斯", "Napoli": "那不勒斯", 
    "Roma": "罗马", "Lazio": "拉齐奥", "Atalanta": "亚特兰大", "Fiorentina": "佛罗伦萨", "Bologna": "博洛尼亚",
    "Torino": "都灵", "Udinese": "乌迪内斯", "Genoa": "热那亚", "Cagliari": "卡利亚里", "Lecce": "莱切",
    "Verona": "维罗纳", "Empoli": "恩波利", "Parma": "帕尔马", "Como": "科莫", "Venezia": "威尼斯", "Monza": "蒙扎",
    "Paris SG": "巴黎圣日耳曼", "Monaco": "摩纳哥", "Marseille": "马赛", "Lille": "里尔", "Lyon": "里昂",
    "Nice": "尼斯", "Lens": "朗斯", "Rennes": "雷恩", "Strasbourg": "斯特拉斯堡", "Brest": "布雷斯特",
    "Toulouse": "图卢兹", "Montpellier": "蒙彼利埃", "Nantes": "南特", "Reims": "兰斯",
    "Auxerre": "欧塞尔", "Angers": "昂热", "Le Havre": "勒阿弗尔", "St Etienne": "圣埃蒂安", "Metz": "梅斯", "Lorient": "洛里昂", "Paris FC": "巴黎FC"
}
CN_TO_EN = {v: k for k, v in EN_TO_CN.items()}

def translate_to_en(name):
    name = str(name).strip()
    return CN_TO_EN.get(name, name)

def translate_to_cn(name):
    name = str(name).strip()
    return EN_TO_CN.get(name, name)

def _std(df):
    df = df.copy()
    rename_dict = {"HxG": "xg_home", "AxG": "xg_away", 
                   "AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a",
                   "B365H": "b365_h", "B365D": "b365_d", "B365A": "b365_a"}
    df = df.rename(columns=rename_dict)
    
    if "odds_h" not in df.columns and "b365_h" in df.columns:
        df["odds_h"], df["odds_d"], df["odds_a"] = df["b365_h"], df["b365_d"], df["b365_a"]
        
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])
    
    for col in ["xg_home", "xg_away", "odds_h", "odds_d", "odds_a"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
            
    return df[["date", "home", "away", "hg", "ag", "xg_home", "xg_away", "odds_h", "odds_d", "odds_a"]]

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
        raw = raw.rename(columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
        frames.append(_std(raw))
    except Exception as e:
        pass

    if not frames:
        raise RuntimeError(f"{code} 无数据")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last").sort_values("date")
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
            loss = poisson_deviance(df["xg_home"].values, lh) + poisson_deviance(df["xg_away"].values, la)
        else:
            loss = -(poisson.logpmf(df["hg"].values, lh) + poisson.logpmf(df["ag"].values, la))
        return np.sum(w * loss)

    def fit(self, df):
        df_fit_all = df.dropna(subset=["hg", "ag"]).copy()
        teams = sorted(set(df_fit_all["home"]) | set(df_fit_all["away"]))
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        df_fit_all["home_idx"] = df_fit_all["home"].map(self._idx)
        df_fit_all["away_idx"] = df_fit_all["away"].map(self._idx)
        days = (df_fit_all["date"].max() - df_fit_all["date"]).dt.days.values
        w = np.exp(-self.xi * days)

        if self.use_xg:
            mask = (df_fit_all["xg_home"].notna() & df_fit_all["xg_away"].notna()).values
            if mask.sum() / len(df_fit_all) < 0.5:
                self.use_xg = False
            else:
                df_fit = df_fit_all[mask].reset_index(drop=True)
                w_fit = w[mask]
        if not self.use_xg:
            df_fit = df_fit_all.reset_index(drop=True)
            w_fit = w

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.1]])
        res = minimize(self._loss, x0, args=(df_fit, n, w_fit),
                       method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
        self.attack, self.defence, self.home_adv = self._unpack(res.x, n)
        self._fit_rho(df_fit_all, w)
        return self

    def _fit_rho(self, df, w):
        hi, ai = df["home_idx"].values, df["away_idx"].values
        hg, ag = df["hg"].values, df["ag"].values
        lh = np.exp(self.attack[hi] - self.defence[ai] + self.home_adv)
        la = np.exp(self.attack[ai] - self.defence[hi])

        def nll(rho):
            t = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
            t[m00] = 1 - lh[m00] * la[m00] * rho; t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho; t[m11] = 1 - rho
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
        return []
    df = df.dropna(subset=["league", "home", "away"])
    out = []
    for _, r in df.iterrows():
        code = str(r["league"]).strip()
        if code not in LEAGUE_NAMES:
            continue
        h_en = translate_to_en(r["home"])
        a_en = translate_to_en(r["away"])
        out.append((code, h_en, a_en))
    return out

def push_to_wechat(md_text):
    """推送到微信"""
    key = os.environ.get("SERVERCHAN_KEY")
    if not key:
        print("未配置 SERVERCHAN_KEY，跳过微信推送。")
        return
    
    print("正在推送到微信...")
    url = f"https://sctapi.ftqq.com/{key}.send"
    data = {
        "title": f"⚽ 今日足球预测 {datetime.now().strftime('%m月%d日')}",
        "desp": md_text
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            print("微信推送成功！")
        else:
            print(f"微信推送失败: {r.text}")
    except Exception as e:
        print(f"微信推送异常: {e}")

def main():
    print(f"  全自动预测系统 (微信推送版) | {datetime.now():%Y-%m-%d %H:%M}")
    targets = load_fixtures()
    print(f"从 fixtures.csv 读到 {len(targets)} 场比赛")

    if not targets:
        return

    results = []
    cache = {}
    fitted = {}

    for code, h_en, a_en in targets:
        name = LEAGUE_NAMES[code]

        if code not in cache:
            try:
                cache[code] = load_league(code)
            except Exception:
                cache[code] = None
                
        if cache[code] is None:
            continue

        if code not in fitted:
            print(f"[{name}] 正在拟合模型 ...")
            fitted[code] = DixonColes(use_xg=True).fit(cache[code])

        model = fitted[code]
        
        h_display = translate_to_cn(h_en)
        a_display = translate_to_cn(a_en)

        if h_en not in model._idx or a_en not in model._idx:
            continue

        g, lh, la = model.predict(h_en, a_en)
        model_hw = float(np.tril(g, -1).sum())
        model_dr = float(np.trace(g))
        model_aw = float(np.triu(g, 1).sum())

        df_league = cache[code]
        match_rows = df_league[(df_league["home"] == h_en) & (df_league["away"] == a_en)]
        
        final_hw, final_dr, final_aw = model_hw, model_dr, model_aw
        
        if len(match_rows) > 0:
            latest_match = match_rows.iloc[-1]
            oh, od, oa = latest_match["odds_h"], latest_match["odds_d"], latest_match["odds_a"]
            
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                imp_h, imp_d, imp_a = 1/oh, 1/od, 1/oa
                total_imp = imp_h + imp_d + imp_a
                mkt_hw, mkt_dr, mkt_aw = imp_h / total_imp, imp_d / total_imp, imp_a / total_imp
                
                final_hw = MARKET_WEIGHT * mkt_hw + (1 - MARKET_WEIGHT) * model_hw
                final_dr = MARKET_WEIGHT * mkt_dr + (1 - MARKET_WEIGHT) * model_dr
                final_aw = MARKET_WEIGHT * mkt_aw + (1 - MARKET_WEIGHT) * model_aw
                
                scale_h = final_hw / model_hw if model_hw > 0 else 1
                scale_d = final_dr / model_dr if model_dr > 0 else 1
                scale_a = final_aw / model_aw if model_aw > 0 else 1
                
                for h in range(g.shape[0]):
                    for a in range(g.shape[1]):
                        if h > a: g[h, a] *= scale_h
                        elif h == a: g[h, a] *= scale_d
                        else: g[h, a] *= scale_a
                g /= g.sum()

        over_25 = 0.0
        btts_yes = 0.0
        for h in range(g.shape[0]):
            for a in range(g.shape[1]):
                if h + a >= 3: over_25 += g[h, a]
                if h > 0 and a > 0: btts_yes += g[h, a]
        
        under_25 = 1.0 - over_25
        btts_no = 1.0 - btts_yes

        scores = [(f"{h}-{a}", float(g[h, a])) for h in range(g.shape[0]) for a in range(g.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "league": name, 
            "home": h_display, 
            "away": a_display,
            "home_win": round(final_hw, 4), "draw": round(final_dr, 4), "away_win": round(final_aw, 4),
            "over_2_5": round(over_25, 4), "under_2_5": round(under_25, 4),
            "btts_yes": round(btts_yes, 4), "btts_no": round(btts_no, 4),
            "top_scores": "; ".join(f"{s}:{p:.3f}" for s, p in scores[:5]),
        })

    today = datetime.now().strftime("%Y-%m-%d")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"predictions_{today}.csv", index=False, encoding="utf-8-sig")
        print(f"已写入 predictions_{today}.csv")

        # ===== 生成好看的 Markdown 发给微信 =====
        md = f"## ⚽ 今日模型预测 ({datetime.now().strftime('%m-%d')})\n\n"
        for _, row in df.iterrows():
            md += f"### [{row['league']}] {row['home']} vs {row['away']}\n"
            md += f"- **胜平负**: 主 {row['home_win']:.1%} | 平 {row['draw']:.1%} | 客 {row['away_win']:.1%}\n"
            md += f"- **大小球**: 大 {row['over_2_5']:.1%} | 小 {row['under_2_5']:.1%}\n"
            md += f"- **双方进球**: 是 {row['btts_yes']:.1%} | 否 {row['btts_no']:.1%}\n"
            md += f"- **推荐比分**: `{row['top_scores'].split(';')[0]}`\n\n"
        
        push_to_wechat(md)

if __name__ == "__main__":
    main()
