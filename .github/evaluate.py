import glob
import io
import requests
import pandas as pd
from datetime import datetime

# 支持复盘的联赛（包含五大联赛+英冠）
LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "E1"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/2627/{}.csv"

def load_dict():
    """加载中文字典，用于将预测文件里的中文名翻译回英文去匹配赛果"""
    try:
        df = pd.read_csv("teams_cn.csv", encoding="utf-8-sig")
        return dict(zip(df["en"].str.strip(), df["cn"].str.strip()))
    except Exception:
        return {}

EN_TO_CN = load_dict()

def get_actual_results():
    """去网上拉取最新的赛果"""
    actual_matches = {}
    for lg in LEAGUES:
        try:
            url = BASE_URL.format(lg)
            r = requests.get(url, timeout=30)
            text = r.content.decode("utf-8-sig", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            
            for _, row in df.iterrows():
                # 只要比赛有比分，就说明踢完了
                if pd.notna(row.get("FTHG")) and pd.notna(row.get("FTAG")):
                    h = str(row["HomeTeam"]).strip()
                    a = str(row["AwayTeam"]).strip()
                    actual_matches[f"{h}||{a}"] = (int(row["FTHG"]), int(row["FTAG"]))
        except Exception:
            pass
    return actual_matches

def main():
    actual = get_actual_results()
    if not actual:
        print("没拉到任何赛果，可能是网络问题或数据源未更新。")
        return

    # 找到所有预测文件
    pred_files = sorted(glob.glob("predictions_*.csv"))
    
    hits = 0
    total = 0
    details = []

    for f in pred_files:
        date_str = f.replace("predictions_", "").replace(".csv", "")
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "message" in df.columns: continue
            
            for _, row in df.iterrows():
                h_cn = str(row["home"]).strip()
                a_cn = str(row["away"]).strip()
                top_scores = str(row["top_scores"]).split(";")[0].split(":")[0].strip()
                
                # 将中文名反查为英文名
                h_en, a_en = h_cn, a_cn
                for en, cn in EN_TO_CN.items():
                    if cn == h_cn: h_en = en
                    if cn == a_cn: a_en = en

                key = f"{h_en}||{a_en}"
                if key in actual:
                    actual_h, actual_a = actual[key]
                    pred_h, pred_a = map(int, top_scores.split("-"))
                    
                    act_res = "H" if actual_h > actual_a else ("D" if actual_h == actual_a else "A")
                    pred_res = "H" if pred_h > pred_a else ("D" if pred_h == pred_a else "A")
                    
                    total += 1
                    res_ok = (act_res == pred_res)
                    score_ok = (pred_h == actual_h and pred_a == actual_a)
                    if res_ok: hits += 1
                    
                    details.append(f"| {date_str} | {h_cn} vs {a_cn} | {top_scores} | {actual_h}-{actual_a} | {'✅' if res_ok else '❌'} | {'✅' if score_ok else '❌'} |")
        except Exception:
            pass

    # 生成 Markdown 格式的报表
    md = f"# 📊 模型历史战绩 (更新于 {datetime.now().strftime('%Y-%m-%d')})\n\n"
    if total > 0:
        md += f"- **已结算总场次**: {total} 场\n"
        md += f"- **胜平负方向命中率**: {hits/total:.1%}\n\n"
        md += "| 预测日期 | 比赛 | 预测最高比分 | 实际比分 | 胜负对错 | 比分对错 |\n|---|---|---|---|---|---|\n"
        md += "\n".join(details)
    else:
        md += "暂无已完赛的预测记录（或者预测的比赛还没踢）。"

    with open("performance.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("已生成 performance.md")

if __name__ == "__main__":
    main()
