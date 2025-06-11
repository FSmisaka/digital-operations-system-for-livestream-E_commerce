import matplotlib
matplotlib.use('Agg')  # 在导入 plt 之前设置后端
import pandas as pd 
import matplotlib.pyplot as plt
import json
import seaborn as sns
from plotly import graph_objects as go
from datetime import timedelta
import matplotlib.font_manager as fm
import plotly.express as px
import numpy as np
import os
import base64
from io import BytesIO
from flask import current_app
from views.data_utils import load_data as load_data_

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    raw_data = pd.read_csv("live_stream_raw_data2.csv", parse_dates=["timestamp"])
    with open("products.json", "r") as f:
        products = json.load(f)
    return raw_data, products

def preprocess_data(raw_data, products):
    raw_data["time_minute"] = raw_data["timestamp"].dt.floor("min")
    product_map = {p["product_id"]: p["name"] for p in products}
    product_map[0] = "无"
    raw_data["product_name"] = raw_data["product_id"].map(product_map)
    return raw_data

def analyze_and_visualize(raw_data):
    fig = plt.figure(figsize=(20, 20))

    # 图1：直播间实时活动趋势（进入、离开、实时人数、弹幕）
    ax1 = plt.subplot(3, 2, 1)
    enter_leave = raw_data[raw_data["event_type"].isin(["enter", "leave"])].groupby(["time_minute", "event_type"]).size().unstack(fill_value=0)
    message_count = raw_data[raw_data["event_type"] == "message"].groupby("time_minute").size()
    
    realtime = enter_leave.copy()
    realtime["人数"] = (enter_leave["enter"].cumsum() - enter_leave["leave"].cumsum())
    realtime["弹幕数"] = message_count
    realtime = realtime.fillna(0)

    realtime[["enter", "leave", "人数", "弹幕数"]].plot(ax=ax1, linewidth=2)
    ax1.set_title("直播间实时活动趋势", fontsize=12, fontweight="bold")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("数量")
    ax1.legend(["进入数", "离开数", "实时人数", "弹幕数"], title="事件类型")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # 图2：商品销量柱状图（取消转化率）
    ax2 = plt.subplot(3, 2, 2)
    product_sales = raw_data[(raw_data["event_type"] == "purchase") & (raw_data["product_id"] > 0)] \
        .groupby("product_name").size()
    product_sales.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title("商品销量", fontsize=12, fontweight="bold")
    ax2.set_xlabel("商品名称")
    ax2.set_ylabel("销量")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.7)

    # 图3：商品销售额饼图
    ax3 = plt.subplot(3, 2, 3)
    sales_amount = raw_data[raw_data["event_type"] == "purchase"].groupby("product_name")["amount"].sum()
    sales_amount.plot.pie(ax=ax3, autopct='%1.1f%%', startangle=90, 
                         colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax3.set_title("商品销售额占比", fontsize=12, fontweight="bold")
    ax3.set_ylabel("")

    # 图4：用户留存曲线（每分钟：1 - 离开数/进入数）
    ax4 = plt.subplot(3, 2, 4)
    minute_data = raw_data[raw_data["event_type"].isin(["enter", "leave"])].groupby(["time_minute", "event_type"]).size().unstack(fill_value=0)
    minute_data["留存率"] = 1 - minute_data["leave"] / minute_data["enter"]
    minute_data["留存率"] = minute_data["留存率"].clip(lower=0, upper=1).fillna(0)
    ax4.plot(minute_data.index, minute_data["留存率"], marker='o', linestyle='-', color='purple')
    ax4.set_title("用户留存曲线（每分钟）", fontsize=12, fontweight="bold")
    ax4.set_xlabel("时间")
    ax4.set_ylabel("留存率")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.set_ylim(0, 1)

    # 图5：用户转化漏斗图（只保留点击→下单→支付）
    funnel_data = {
        "阶段": ["点击", "下单", "支付"],
        "人数": [
            raw_data["event_type"].eq("click").sum(),
            raw_data["event_type"].eq("purchase").sum(),
            raw_data["event_type"].eq("payment").sum()
        ]
    }

    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data["阶段"],
        x=funnel_data["人数"],
        textinfo="value+percent previous",
        opacity=0.8,
        marker={"color": ["#EF553B", "#00CC96", "#AB63FA"]},
        connector={"line": {"color": "royalblue", "width": 2}}
    ))

    fig_funnel.update_layout(
        title="用户转化漏斗分析（点击→下单→支付）",
        margin=dict(l=100, r=100, t=100, b=100),
        font=dict(family="SimHei", size=12)
    )

    plt.tight_layout()
    plt.savefig("matplotlib_all_plots.png", dpi=300, bbox_inches="tight")
    plt.show()
    fig_funnel.show()

    return {
        "conversion_funnel": funnel_data
    }

def get_live_analysis(live_number):
    try:
        # 确保关闭所有之前的图形
        plt.close('all')
        
        # 加载数据
        data_path = os.path.join(current_app.root_path, 'data', 'live')
        raw_data = pd.read_csv(os.path.join(data_path, f'live_stream_raw_data{live_number}.csv'), parse_dates=["timestamp"])
        
        # 预处理数据 - 只需要时间处理
        raw_data["time_minute"] = raw_data["timestamp"].dt.floor("min")

        # 创建一个字典来存储所有图表的base64编码
        plots = {}

        # 为每个图表创建单独的figure
        for plot_type in ['activity_trend', 'product_sales', 'sales_pie', 'retention_curve', 'conversion_funnel']:
            plt.figure(figsize=(10, 9))
            
            if plot_type == 'activity_trend':
                # 活动趋势图
                enter_leave = raw_data[raw_data["event_type"].isin(["enter", "leave"])].groupby(["time_minute", "event_type"]).size().unstack(fill_value=0)
                message_count = raw_data[raw_data["event_type"] == "message"].groupby("time_minute").size()
                realtime = enter_leave.copy()
                realtime["人数"] = (enter_leave["enter"].cumsum() - enter_leave["leave"].cumsum())
                realtime["弹幕数"] = message_count
                realtime = realtime.fillna(0)
                realtime[["enter", "leave", "人数", "弹幕数"]].plot(linewidth=2)
                plt.title("直播间实时活动趋势")
                plt.xlabel("时间")
                plt.ylabel("数量")
                plt.legend(["进入数", "离开数", "实时人数", "弹幕数"])
                plt.grid(True, linestyle="--", alpha=0.7)
            
            elif plot_type == 'product_sales':
                # 商品销量图
                product_sales = raw_data[raw_data["event_type"] == "purchase"].groupby("product_id").size()
                product_sales.plot(kind='bar', color='skyblue')
                plt.title("商品销量")
                plt.xlabel("商品ID")
                plt.ylabel("销量")
                plt.xticks(rotation=45)
                plt.grid(True, axis='y', linestyle="--", alpha=0.7)
            
            elif plot_type == 'sales_pie':
                # 销售额饼图
                products = load_data_('../data/forum/topics.json')
                products = [p for p in products if p['category'] != 'announcement']
                product_names = {str(topic['id']): topic['title'] for topic in products if 'id' in topic.keys()}

                raw_data["product_id"] = raw_data["product_id"]+2
                sales_amount = raw_data[raw_data["event_type"] == "purchase"].groupby("product_id")["amount"].sum()
                labels = [product_names.get(str(idx), f'商品{idx}') for idx in sales_amount.index]
                sizes = sales_amount.values
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))

                if len(sizes) > 8:
                    labels = labels[:7] + ['其他']
                    sizes = np.concatenate([sizes[:7], [sizes[7:].sum()]])
                    colors = colors[:8]
                
                plt.pie(sizes, 
                        labels=labels, 
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=colors)
                
                plt.title("商品销售额占比")
            
            elif plot_type == 'retention_curve':
                # 留存曲线
                minute_data = raw_data[raw_data["event_type"].isin(["enter", "leave"])].groupby(["time_minute", "event_type"]).size().unstack(fill_value=0)
                minute_data["留存率"] = 1 - minute_data["leave"] / minute_data["enter"]
                minute_data["留存率"] = minute_data["留存率"].clip(lower=0, upper=1).fillna(0)
                plt.plot(minute_data.index, minute_data["留存率"], marker='o', linestyle='-', color='purple')
                plt.title("用户留存曲线（每分钟）")
                plt.xlabel("时间")
                plt.ylabel("留存率")
                plt.grid(True, linestyle="--", alpha=0.7)
            
            elif plot_type == 'conversion_funnel':
                # 转化漏斗
                funnel_data = {
                    "阶段": ["点击", "下单", "支付"],
                    "人数": [
                        raw_data["event_type"].eq("click").sum(),
                        raw_data["event_type"].eq("purchase").sum(),
                        raw_data["event_type"].eq("payment").sum()
                    ]
                }
                plt.bar(funnel_data["阶段"], funnel_data["人数"])
                plt.title("用户转化漏斗")
                plt.ylabel("人数")
                for i, v in enumerate(funnel_data["人数"]):
                    plt.text(i, v, str(v), ha='center', va='bottom')

            # 将图表转换为base64字符串
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plots[plot_type] = base64.b64encode(buffer.getvalue()).decode()
            plt.close('all')  # Close all figures

        return {'plots': plots}

    except Exception as e:
        import logging
        logging.error(f'生成直播分析时出错: {str(e)}')
        return {"error": str(e)}

if __name__ == "__main__":
    raw_data, products = load_data()
    raw_data = preprocess_data(raw_data, products)
    metrics = analyze_and_visualize(raw_data, products)

    print("\n转化漏斗:")
    for i in range(1, len(metrics['conversion_funnel']['阶段'])):
        prev = metrics['conversion_funnel']['人数'][i-1]
        curr = metrics['conversion_funnel']['人数'][i]
        print(f"- {metrics['conversion_funnel']['阶段'][i-1]}→{metrics['conversion_funnel']['阶段'][i]}: {curr/prev:.1%}" if prev else "- 无法计算")

