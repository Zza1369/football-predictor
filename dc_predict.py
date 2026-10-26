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
MARKET_WEIGHT = 0.5
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
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
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
    """生成历史战绩报告"""
    pred_files = sorted(glob.glob("predictions_*.csv"))
    hits, total = 0, 0
    details = []
    
    for f in pred_files:
        date_str = f.replace("predictions_", "").replace(".csv", "")
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "message" in df.columns: continue
            for _, row in df.iterrows():
                h_cn, a_cn = str(row["home"]).strip(), str(row["away"]).strip()
                top_score = str(row["top_scores"]).split(";")[0].split(":")[0].strip()
                
                h_en, a_en = translate_to_en(h_cn), translate_to_en(a_cn)
                key = f"{h_en}||{a_en}"
                
                if key in actual_results:
                    actual_h, actual_a = actual_results[key]
                    pred_h, pred_a = map(int, top_score.split("-"))
                    
                    act_res = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
                    pred_res = "H" if pred_h > pred_a else ("D" if pred_h == pred_a else "A")
                    
                    total += 1
                    res_ok = (act_res == pred_res)
                    score_ok = (pred_h == actual_h and pred_a == actual_a)
                    if res_ok: hits += 1
                    
                    details.append(f"| {date_str} | {h_cn} vs {a_cn} | {top_score} | {actual_h}-{actual_a} | {'✅' if res_ok else '❌'} | {'✅' if score_ok else '❌'} |")
        except: pass

    md = f"### 📊 历史战绩回顾\n"
    if total > 0:
        md += f"**已结算 {total} 场 | 胜平负命中率: {hits/total:.1%}**\n\n"
        md += "| 日期 | 比赛 | 预测比分 | 实际比分 | 胜负 | 比分 |\n|---|---|---|---|---|---|\n"
        md += "\n".join(details[-10:]) # 微信里最多显示最近10场，免得刷屏
        if len(details) > 10: md += f"\n\n*(仅显示最近 10 场)*\n"
    else:
        md += "暂无已完赛记录（或比赛还没踢）。\n"
    return md

def push_to_wechat(md_text):
    key = os.environ.get("SERVERCHAN_KEY")
    if not key: return
    print("正在推送到微信...")
    try:
        requests.post(f"https://sctapi.ftqq.com/{key}.send", data={"title": f"⚽ 今日预测与战绩 {datetime.now().strftime('%m月%d日')}", "desp": md_text}, timeout=10)
        print("微信推送成功！")
    except Exception as e:
        print(f"微信推送异常: {e}")

def main():
    print(f"  终极量化预测系统 | {datetime.now():%Y-%m-%d %H:%M}")
    
    try:
        fixtures = pd.read_csv(FIXTURES_CSV, encoding="utf-8-sig").dropna(subset=["league", "home", "away"])
    except:
        return

    results, cache, fitted = [], {}, {}
    
    # 1. 先拉取所有赛果，用于复盘
    actual_results = {}
    for code in LEAGUE_NAMES.keys():
        try:
            df = load_league(code)
            for _, r in df.dropna(subset=["hg", "ag"]).iterrows():
                actual_results[f"{r['home']}||{r['away']}"] = (int(r['hg']), int(r['ag']))
        except: pass

    # 2. 生成今日预测报告
    md = f"## ⚽ 今日模型预测 ({datetime.now().strftime('%m-%d')})\n\n"
    if len(fixtures) == 0:
        md += "今日没有填写的预测比赛。\n\n"

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
        today = pd.Timestamp.now()
        def get_rest(team):
            m = df_league[((df_league["home"] == team) | (df_league["away"] == team)) & (df_league["date"] < today)]
            return 99 if len(m)==0 else (today - m["date"].max()).days
        rest_h, rest_a = get_rest(h_en), get_rest(a_en)

        g, lh, la = model.predict(h_en, a_en, rest_h, rest_a)
        model_hw = float(np.tril(g, -1).sum())
        model_dr = float(np.trace(g))
        model_aw = float(np.triu(g, 1).sum())

        final_hw, final_dr, final_aw = model_hw, model_dr, model_aw
        match_rows = df_league[(df_league["home"] == h_en) & (df_league["away"] == a_en)]
        
        val_flag_h, val_flag_a = "", ""
        if len(match_rows) > 0:
            oh, od, oa = match_rows.iloc[-1]["odds_h"], match_rows.iloc[-1]["odds_d"], match_rows.iloc[-1]["odds_a"]
            if pd.notna(oh) and pd.notna(od) and pd.notna(oa):
                imp_h, imp_d, imp_a = 1/oh, 1/od, 1/oa
                tot = imp_h + imp_d + imp_a
                mkt_hw, mkt_dr, mkt_aw = imp_h/tot, imp_d/tot, imp_a/tot
                if model_hw - mkt_hw > VALUE_THRESHOLD: val_flag_h = "🔥 主胜有价值"
                if model_aw - mkt_aw > VALUE_THRESHOLD: val_flag_a = "🔥 客胜有价值"

                final_hw = MARKET_WEIGHT * mkt_hw + (1 - MARKET_WEIGHT) * model_hw
                final_dr = MARKET_WEIGHT * mkt_dr + (1 - MARKET_WEIGHT) * model_dr
                final_aw = MARKET_WEIGHT * mkt_aw + (1 - MARKET_WEIGHT) * model_aw
                
                scale_h, scale_d, scale_a = final_hw/model_hw, final_dr/model_dr, final_aw/model_aw
                for h in range(g.shape[0]):
                    for a in range(g.shape[1]):
                        if h > a: g[h, a] *= scale_h
                        elif h == a: g[h, a] *= scale_d
                        else: g[h, a] *= scale_a
                g /= g.sum()

        ah_win = sum(g[h,a] for h in range(9) for a in range(9) if h-a >= 2)
        ah_push = sum(g[h,a] for h in range(9) for a in range(9) if h-a == 1)
        over_25 = sum(g[h,a] for h in range(9) for a in range(9) if h+a >= 3)
        btts_yes = sum(g[h,a] for h in range(9) for a in range(9) if h>0 and a>0)

        # 取前3个比分
        scores = sorted([(f"{h}-{a}", float(g[h, a])) for h in range(g.shape[0]) for a in range(g.shape[1])], key=lambda x: x[1], reverse=True)
        top3_str = "、".join([f"**{s[0]}**({s[1]:.1%})" for s in scores[:3]])

        results.append({
            "league": name, "home": h_cn, "away": a_cn,
            "home_win": round(final_hw, 4), "draw": round(final_dr, 4), "away_win": round(final_aw, 4),
            "top_scores": "; ".join(f"{s}:{p:.3f}" for s, p in scores[:5]),
        })
        
        md += f"### [{name}] {h_cn} vs {a_cn}\n"
        if val_flag_h or val_flag_a: md += f"> 💡 **价值提示**: {val_flag_h} {val_flag_a}\n"
        if rest_h < 4: md += f"> ⚠️ {h_cn} 休息不足(仅{rest_h}天)\n"
        if rest_a < 4: md += f"> ⚠️ {a_cn} 休息不足(仅{rest_a}天)\n"
        md += f"- **胜平负**: 主 {final_hw:.1%} | 平 {final_dr:.1%} | 客 {final_aw:.1%}\n"
        md += f"- **主让一球(-1)**: 赢盘 {ah_win:.1%} | 走水 {ah_push:.1%} | 输盘 {1-ah_win-ah_push:.1%}\n"
        md += f"- **大小/双方进球**: 大 {over_25:.1%} | 双方进 {btts_yes:.1%}\n"
        md += f"- **推荐比分**: {top3_str}\n\n"

    # 3. 写入文件
    if results:
        today_str = datetime.now().strftime("%Y-%m-%d")
        pd.DataFrame(results).to_csv(f"predictions_{today_str}.csv", index=False, encoding="utf-8-sig")

    # 4. 追加复盘战绩并推送
    md += "---\n"
    md += get_performance_report(actual_results)
    
    with open("performance.md", "w", encoding="utf-8") as f:
        f.write(md)
        
    push_to_wechat(md)

if __name__ == "__main__":
    main()
