import io
import os
import json
import glob
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
WEIGHT_A = 0.5 
WEIGHT_B = 0.8 
VALUE_THRESHOLD = 0.05

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

def translate_to_en(name): return CN_TO_EN.get(str(name).strip(), str(name).strip())
def translate_to_cn(name): return EN_TO_CN.get(str(name).strip(), str(name).strip())

def _std(df):
    df = df.copy()
    rename_dict = {"HxG": "xg_home", "AxG": "xg_away", "AvgH": "odds_h", "AvgD": "odds_d", "AvgA": "odds_a", "B365H": "b365_h", "B365D": "b365_d", "B365A": "b365_a"}
    df = df.rename(columns=rename_dict)
    if "odds_h" not in df.columns and "b365_h" in df.columns: df["odds_h"], df["odds_d"], df["odds_a"] = df["b365_h"], df["b365_d"], df["b365_a"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    for col in ["xg_home", "xg_away", "odds_h", "odds_d", "odds_a"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        else: df[col] = np.nan
    return df[["date", "home", "away", "hg", "ag", "xg_home", "xg_away", "odds_h", "odds_d", "odds_a"]]

def load_league(code):
    frames = []
    try:
        local = pd.read_csv(LOCAL_CSV, encoding="utf-8-sig")
        local = local[local["league"] == LEAGUE_NAMES[code]].copy()
        if len(local): frames.append(_std(local))
    except: pass
    url = BASE_URL.format(season=SEASON_2627, league=code)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig", errors="replace")))
        raw = raw.rename(columns={"Date": "date", "HomeTeam": "home", "AwayTeam": "away", "FTHG": "hg", "FTAG": "ag"})
        frames.append(_std(raw))
    except: pass
    if not frames: raise RuntimeError(f"{code} 无数据")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["date", "home", "away"], keep="last").sort_values("date").reset_index(drop=True)

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
        return p[:n], p[n:2*n], p[2*n:3*n], p[3*n:4*n], p[4*n]

    def _loss(self, p, df, n, w):
        ah, aa, dh, da, h_adv = self._unpack(p, n)
        hi, ai = df["home_idx"].values, df["away_idx"].values
        lh = np.exp(ah[hi] - da[ai] + h_adv)
        la = np.exp(aa[ai] - dh[hi])
        if self.use_xg: loss = poisson_deviance(df["xg_home"].values, lh) + poisson_deviance(df["xg_away"].values, la)
        else: loss = -(poisson.logpmf(df["hg"].values, lh) + poisson.logpmf(df["ag"].values, la))
        reg = 0.1 * np.sum(p[:4*n]**2)
        return np.sum(w * loss) + reg

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
            if mask.sum() / len(df_fit_all) < 0.5: self.use_xg = False
            else:
                df_fit = df_fit_all[mask].reset_index(drop=True)
                w_fit = w[mask]
        if not self.use_xg:
            df_fit = df_fit_all.reset_index(drop=True)
            w_fit = w
        x0 = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), [0.1]])
        res = minimize(self._loss, x0, args=(df_fit, n, w_fit), method="L-BFGS-B", options={"maxiter": 1000})
        self.ah, self.aa, self.dh, self.da, self.home_adv = self._unpack(res.x, n)
        self._fit_rho(df_fit_all, w)
        return self

    def _fit_rho(self, df, w):
        hi, ai = df["home_idx"].values, df["away_idx"].values
        hg, ag = df["hg"].values, df["ag"].values
        lh = np.exp(self.ah[hi] - self.da[ai] + self.home_adv)
        la = np.exp(self.aa[ai] - self.dh[hi])
        def nll(rho):
            t = np.ones_like(lh)
            m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1); m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)
            t[m00] = 1 - lh[m00] * la[m00] * rho; t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho; t[m11] = 1 - rho
            t = np.clip(t, 1e-10, None)
            ll = poisson.logpmf(hg, lh) + poisson.logpmf(ag, la) + np.log(t)
            return -np.sum(w * ll)
        r = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(r.x)

    def predict(self, home, away, rest_h=99, rest_a=99, max_goals=MAX_GOALS):
        hi, ai = self._idx[home], self._idx[away]
        lh = np.exp(self.ah[hi] - self.da[ai] + self.home_adv)
        la = np.exp(self.aa[ai] - self.dh[hi])
        if rest_h < 4: lh *= 0.9
        if rest_a < 4: la *= 0.9
        g = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lh) * poisson.pmf(a, la)
                p *= rho_correction(h, a, lh, la, self.rho)
                g[h, a] = p
        g = np.clip(g, 0, None)
        g /= g.sum()
        return g, lh, la

def get_performance_report(actual_results):
    pred_files = sorted(glob.glob("predictions_*.csv"))
    stats = {"total": 0, "A_wl": 0, "B_wl": 0, "A_score": 0, "B_score": 0}
    details = []
    
    for f in pred_files:
        date_str = f.replace("predictions_", "").replace(".csv", "")
        try:
            pred_date = pd.to_datetime(date_str).normalize()
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "message" in df.columns: continue
            
            for _, row in df.iterrows():
                h_cn, a_cn = str(row["home"]).strip(), str(row["away"]).strip()
                h_en, a_en = translate_to_en(h_cn), translate_to_en(a_cn)
                key = f"{h_en}||{a_en}"
                
                if key in actual_results:
                    matched_match = None
                    for match in actual_results[key]:
                        if match["date"] >= pred_date:
                            matched_match = match
                            break
                    
                    if matched_match:
                        actual_h, actual_a = matched_match["hg"], matched_match["ag"]
                        act_res = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
                        
                        top_a = str(row.get("top_scores_A", row.get("top_scores", ""))).split(";")[0].split(":")[0].strip()
                        pred_h_a, pred_a_a = map(int, top_a.split("-"))
                        pred_res_a = "H" if pred_h_a > pred_a_a else ("D" if pred_h_a == pred_a_a else "A")
                        A_wl_ok = (act_res == pred_res_a)
                        A_score_ok = (pred_h_a == actual_h and pred_a_a == actual_a)
                        
                        top_b = str(row.get("top_scores_B", row.get("top_scores", ""))).split(";")[0].split(":")[0].strip()
                        pred_h_b, pred_a_b = map(int, top_b.split("-"))
                        pred_res_b = "H" if pred_h_b > pred_a_b else ("D" if pred_h_b == pred_a_b else "A")
                        B_wl_ok = (act_res == pred_res_b)
                        B_score_ok = (pred_h_b == actual_h and pred_a_b == actual_a)

                        stats["total"] += 1
                        if A_wl_ok: stats["A_wl"] += 1
                        if B_wl_ok: stats["B_wl"] += 1
                        if A_score_ok: stats["A_score"] += 1
                        if B_score_ok: stats["B_score"] += 1
                        
                        details.append({
                            "date": date_str, "teams": f"{h_cn} vs {a_cn}",
                            "act": f"{actual_h}-{actual_a}",
                            "pred_a": top_a, "pred_b": top_b,
                            "a_wl": A_wl_ok, "b_wl": B_wl_ok,
                            "a_sc": A_score_ok, "b_sc": B_score_ok
                        })
        except Exception: pass

    md = f"### 📊 历史战绩对比 (A: 50%赔率 | B: 80%赔率)\n\n"
    if stats["total"] > 0:
        t = stats["total"]
        md += f"**已结算 {t} 场**\n"
        md += f"- 🎯 胜平负: 🅰️ **{stats['A_wl']/t:.1%}**  |  🅱️ **{stats['B_wl']/t:.1%}**\n"
        md += f"- ⚽ 比分: 🅰️ **{stats['A_score']/t:.1%}**  |  🅱️ **{stats['B_score']/t:.1%}**\n\n"
        
        md += "**近期明细 (最近5场):**\n"
        for d in details[-5:]:
            md += f"⚔️ **{d['teams']}** ({d['date'][5:]}) | 实际 `{d['act']}`\n"
            md += f"🅰️ 预测 `{d['pred_a']}` └ 胜负 {'✅' if d['a_wl'] else '❌'} | 比分 {'✅' if d['a_sc'] else '❌'}\n"
            md += f"🅱️ 预测 `{d['pred_b']}` └ 胜负 {'✅' if d['b_wl'] else '❌'} | 比分 {'✅' if d['b_sc'] else '❌'}\n\n"
    else:
        md += "暂无已完赛记录。\n"
    return md

def push_to_wechat(md_text):
    key = os.environ.get("SERVERCHAN_KEY")
    if not key: return
    try:
        requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": f"⚽ 双模型预测与战绩 {datetime.now().strftime('%m月%d日')}", "desp": md_text}, timeout=10)
    except Exception: pass

def main():
    try:
        fixtures = pd.read_csv(FIXTURES_CSV, encoding="utf-8-sig").dropna(subset=["league", "home", "away"])
    except: return

    results, cache, fitted = [], {}, {}
    actual_results = {}
    
    for code in LEAGUE_NAMES.keys():
        try:
            df = load_league(code)
            for _, r in df.dropna(subset=["hg", "ag"]).iterrows():
                key = f"{r['home']}||{r['away']}"
                if key not in actual_results: actual_results[key] = []
                actual_results[key].append({"date": r['date'], "hg": int(r['hg']), "ag": int(r['ag'])})
        except: pass

    md = f"## ⚽ 今日双模型预测 ({datetime.now().strftime('%m-%d')})\n"
    md += "*(🅰️: 50%赔率融合 | 🅱️: 80%赔率融合)*\n\n"

    for _, r in fixtures.iterrows():
        code = str(r["league"]).strip()
        if code not in LEAGUE_NAMES: continue
        h_en, a_en = translate_to_en(r["home"]), translate_to_en(r["away"])
        h_cn, a_cn = translate_to_cn(h_en), translate_to_cn(a_en)
        name = LEAGUE_NAMES[code]

        if code not in cache:
            try: cache[code] = load_league(code)
            except: cache[code] = None
        if cache[code] is None: continue

        if code not in fitted:
            fitted[code] = DixonColes().fit(cache[code])
        model = fitted[code]

        if h_en not in model._idx or a_en not in model._idx: continue

        df_league = cache[code]
        today = pd.Timestamp.now().normalize()
        def get_rest(team):
            m = df_league[((df_league["home"] == team) | (df_league["away"] == team)) & (df_league["date"] < today)]
            return 99 if len(m)==0 else (today - m["date"].max()).days
        rest_h, rest_a = get_rest(h_en), get_rest(a_en)

        g, lh, la = model.predict(h_en, a_en, rest_h, rest_a)
        model_hw = float(np.tril(g, -1).sum())
        model_dr = float(np.trace(g))
        model_aw = float(np.triu(g, 1).sum())

        match_rows = df_league[(df_league["home"] == h_en) & (df_league["away"] == a_en) & (df_league["date"] >= today)]
        if len(match_rows) == 0: match_rows = df_league[(df_league["home"] == h_en) & (df_league["away"] == a_en)]

        has_odds = False
        if len(match_rows) > 0 and pd.notna(match_rows.iloc[-1]["odds_h"]):
            oh, od, oa = match_rows.iloc[-1]["odds_h"], match_rows.iloc[-1]["odds_d"], match_rows.iloc[-1]["odds_a"]
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                has_odds = True
                imp_h, imp_d, imp_a = 1/oh, 1/od, 1/oa
                tot = imp_h + imp_d + imp_a
                mkt_hw, mkt_dr, mkt_aw = imp_h/tot, imp_d/tot, imp_a/tot

                # 模型 A
                A_hw = WEIGHT_A * mkt_hw + (1 - WEIGHT_A) * model_hw
                A_dr = WEIGHT_A * mkt_dr + (1 - WEIGHT_A) * model_dr
                A_aw = WEIGHT_A * mkt_aw + (1 - WEIGHT_A) * model_aw
                g_A = g.copy()
                g_A[np.tril_indices(9, -1)] *= A_hw/model_hw if model_hw > 0 else 1
                g_A[np.diag_indices(9)] *= A_dr/model_dr if model_dr > 0 else 1
                g_A[np.triu_indices(9, 1)] *= A_aw/model_aw if model_aw > 0 else 1
                g_A /= g_A.sum()
                
                # 模型 B
                B_hw = WEIGHT_B * mkt_hw + (1 - WEIGHT_B) * model_hw
                B_dr = WEIGHT_B * mkt_dr + (1 - WEIGHT_B) * model_dr
                B_aw = WEIGHT_B * mkt_aw + (1 - WEIGHT_B) * model_aw
                g_B = g.copy()
                g_B[np.tril_indices(9, -1)] *= B_hw/model_hw if model_hw > 0 else 1
                g_B[np.diag_indices(9)] *= B_dr/model_dr if model_dr > 0 else 1
                g_B[np.triu_indices(9, 1)] *= B_aw/model_aw if model_aw > 0 else 1
                g_B /= g_B.sum()

        if not has_odds:
            # 没有赔率，退化回纯模型
            A_hw, A_dr, A_aw = model_hw, model_dr, model_aw
            B_hw, B_dr, B_aw = model_hw, model_dr, model_aw
            g_A, g_B = g, g

        scores_A = sorted([(f"{h}-{a}", float(g_A[h, a])) for h in range(g.shape[0]) for a in range(g.shape[1])], key=lambda x: x[1], reverse=True)
        scores_B = sorted([(f"{h}-{a}", float(g_B[h, a])) for h in range(g.shape[0]) for a in range(g.shape[1])], key=lambda x: x[1], reverse=True)

        results.append({
            "league": name, "home": h_cn, "away": a_cn,
            "home_win_A": round(A_hw, 4), "draw_A": round(A_dr, 4), "away_win_A": round(A_aw, 4),
            "top_scores_A": "; ".join(f"{s}:{p:.3f}" for s, p in scores_A[:5]),
            "home_win_B": round(B_hw, 4), "draw_B": round(B_dr, 4), "away_win_B": round(B_aw, 4),
            "top_scores_B": "; ".join(f"{s}:{p:.3f}" for s, p in scores_B[:5]),
        })
        
        ah_win = sum(g_A[h,a] for h in range(9) for a in range(9) if h-a >= 2)
        ah_push = sum(g_A[h,a] for h in range(9) for a in range(9) if h-a == 1)
        ah_lose = 1 - ah_win - ah_push
        over_25 = sum(g_A[h,a] for h in range(9) for a in range(9) if h+a >= 3)
        btts_yes = sum(g_A[h,a] for h in range(9) for a in range(9) if h>0 and a>0)

        md += f"### [{name}] {h_cn} vs {a_cn}\n"
        md += f"- **主让一球(-1)**: 赢盘 {ah_win:.1%} | 走水 {ah_push:.1%} | 输盘 {ah_lose:.1%}\n"
        md += f"- **大小/双方进球**: 大 {over_25:.1%} | 双方进 {btts_yes:.1%}\n"

        if has_odds:
            top3_A = "、".join([f"`{s[0]}`" for s in scores_A[:3]])
            top3_B = "、".join([f"`{s[0]}`" for s in scores_B[:3]])
            md += f"🅰️ **50%赔率**: 主 {A_hw:.1%} | 平 {A_dr:.1%} | 客 {A_aw:.1%}\n"
            md += f"   └ 推荐比分: {top3_A}\n"
            md += f"🅱️ **80%赔率**: 主 {B_hw:.1%} | 平 {B_dr:.1%} | 客 {B_aw:.1%}\n"
            md += f"   └ 推荐比分: {top3_B}\n\n"
        else:
            top3_pure = "、".join([f"`{s[0]}`" for s in scores_A[:3]])
            md += f"💡 **[纯数据]** (暂无实时赔率): 主 {A_hw:.1%} | 平 {A_dr:.1%} | 客 {A_aw:.1%}\n"
            md += f"   └ 推荐比分: {top3_pure}\n\n"

    if results:
        pd.DataFrame(results).to_csv(f"predictions_{datetime.now().strftime('%Y-%m-%d')}.csv", index=False, encoding="utf-8-sig")

    md += "---\n"
    md += get_performance_report(actual_results)
    
    with open("performance.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    push_to_wechat(md)

if __name__ == "__main__":
    main()
