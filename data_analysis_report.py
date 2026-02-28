import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from collections import Counter
import os
import base64
from io import BytesIO

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取清洗后的数据
df = pd.read_csv('D:/workspace/Data_analysis_of_clothing_products/data_clean.csv')

# 创建报告图片存储目录
img_dir = 'report_images'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def fig_to_base64(fig):
    """将 matplotlib 图形转换为 base64 编码的字符串，用于直接嵌入 HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

# 存储每个图表的 base64 字符串
images = {}

# ==================== 数据概览图表 ====================
# 1. 价格分布
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['price'].clip(upper=200), bins=50, edgecolor='black', alpha=0.7)
ax.set_title('价格分布 (≤200)')
ax.set_xlabel('价格')
ax.set_ylabel('频数')
images['price_dist'] = fig_to_base64(fig)
plt.close(fig)

# 2. 评分分布
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['average_rating'], bins=20, edgecolor='black', alpha=0.7, color='orange')
ax.set_title('评分分布')
ax.set_xlabel('平均评分')
ax.set_ylabel('频数')
images['rating_dist'] = fig_to_base64(fig)
plt.close(fig)

# 3. 评论数分布
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['rating_number'].clip(upper=500), bins=50, edgecolor='black', alpha=0.7, color='green')
ax.set_title('评论数分布 (≤500)')
ax.set_xlabel('评论数')
ax.set_ylabel('频数')
images['review_dist'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 类别分析 ====================
category_stats = df.groupby('main_category').agg({
    'title': 'count',
    'price': 'mean',
    'average_rating': 'mean',
    'rating_number': 'sum'
}).rename(columns={
    'title': 'product_count',
    'price': 'avg_price',
    'average_rating': 'avg_rating',
    'rating_number': 'total_reviews'
}).sort_values('product_count', ascending=False).head(10)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# 商品数量
axes[0,0].barh(category_stats.index, category_stats['product_count'])
axes[0,0].set_title('Top10 类别商品数量')
axes[0,0].set_xlabel('商品数')
# 平均价格
axes[0,1].barh(category_stats.index, category_stats['avg_price'], color='orange')
axes[0,1].set_title('Top10 类别平均价格')
axes[0,1].set_xlabel('平均价格 ($)')
# 平均评分
axes[1,0].barh(category_stats.index, category_stats['avg_rating'], color='green')
axes[1,0].set_title('Top10 类别平均评分')
axes[1,0].set_xlabel('平均评分')
axes[1,0].set_xlim(3,5)
# 总评论数
axes[1,1].barh(category_stats.index, category_stats['total_reviews'], color='red')
axes[1,1].set_title('Top10 类别总评论数')
axes[1,1].set_xlabel('总评论数')
plt.tight_layout()
images['category'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 店铺分析 ====================
store_stats = df.groupby('store').agg({
    'title': 'count',
    'rating_number': 'sum',
    'average_rating': 'mean'
}).rename(columns={
    'title': 'product_count',
    'rating_number': 'total_reviews',
    'average_rating': 'avg_rating'
})
top_stores = store_stats.sort_values('total_reviews', ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 商品数量
axes[0].barh(top_stores.index, top_stores['product_count'])
axes[0].set_title('Top10 店铺商品数量')
axes[0].set_xlabel('商品数')
# 总评论数
axes[1].barh(top_stores.index, top_stores['total_reviews'], color='orange')
axes[1].set_title('Top10 店铺总评论数')
axes[1].set_xlabel('总评论数')
plt.tight_layout()
images['store'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 价格与评分关系 ====================
price_bins = [0, 20, 40, 60, 80, 100, 200, 500]
price_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-200', '200-500']
df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=False)
price_rating = df.groupby('price_range', observed=False)['average_rating'].agg(['mean', 'count']).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(price_rating['price_range'], price_rating['mean'], color='skyblue')
axes[0].set_title('各价格区间平均评分')
axes[0].set_xlabel('价格区间 ($)')
axes[0].set_ylabel('平均评分')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(price_rating['price_range'], price_rating['count'], color='orange')
axes[1].set_title('各价格区间商品数量')
axes[1].set_xlabel('价格区间 ($)')
axes[1].set_ylabel('商品数量')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
images['price_rating'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 特征词云 ====================
# 需要先确保 features_list 列存在
if 'features_list' not in df.columns:
    import ast
    def safe_eval_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except:
            return []
    df['features_list'] = df['features'].apply(safe_eval_list)

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

all_features = []
for feat_list in df['features_list']:
    for feat in feat_list:
        if isinstance(feat, str):
            words = feat.lower().split()
            for word in words:
                word_clean = word.strip('.,;:()"\'')
                if word_clean and word_clean not in stop_words and len(word_clean) > 1 and not word_clean.isdigit():
                    all_features.append(word_clean)

feature_counter = Counter(all_features)
wc = wordcloud.WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(feature_counter)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title('商品特征词云')
images['wordcloud'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 视频影响分析 ====================
def has_video_by_title(video_data):
    if pd.isna(video_data):
        return 0
    if isinstance(video_data, list):
        for item in video_data:
            if isinstance(item, dict) and 'title' in item:
                return 1
        return 0
    if isinstance(video_data, str):
        try:
            parsed = ast.literal_eval(video_data)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and 'title' in item:
                        return 1
        except:
            pass
    return 0

df['has_video'] = df['videos'].apply(has_video_by_title)
video_stats = df.groupby('has_video').agg({
    'rating_number': 'mean',
    'average_rating': 'mean',
    'price': 'mean',
    'has_video': 'count'
}).rename(columns={'has_video': 'count'})

fig, axes = plt.subplots(1, 3, figsize=(15,4))
groups = video_stats.index.tolist()
mean_reviews = video_stats['rating_number'].values
mean_ratings = video_stats['average_rating'].values
mean_prices = video_stats['price'].values
x_pos = range(len(groups))
labels = ['无视频' if g==0 else '有视频' for g in groups]

axes[0].bar(x_pos, mean_reviews, color=['gray','gold'][:len(groups)])
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels)
axes[0].set_title('平均评论数对比')
axes[0].set_ylabel('平均评论数')

axes[1].bar(x_pos, mean_ratings, color=['gray','gold'][:len(groups)])
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels)
axes[1].set_title('平均评分对比')
axes[1].set_ylabel('平均评分')

axes[2].bar(x_pos, mean_prices, color=['gray','gold'][:len(groups)])
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels)
axes[2].set_title('平均价格对比')
axes[2].set_ylabel('平均价格')

plt.tight_layout()
images['video'] = fig_to_base64(fig)
plt.close(fig)

# ==================== 生成 HTML 报告 ====================
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>电商数据分析报告</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 30px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }}
        .stat-item {{
            text-align: center;
            min-width: 150px;
            margin: 10px;
        }}
        .stat-number {{
            font-size: 28px;
            font-weight: bold;
            color: #2980b9;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .img-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .footer {{
            margin-top: 40px;
            font-size: 12px;
            color: #aaa;
            text-align: center;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 电商数据分析报告</h1>
    <p>本报告基于 48,955 条服装类商品数据，从多个维度分析商品表现、价格分布、用户反馈等，为业务决策提供数据支持。</p>

    <div class="summary-box">
        <div class="stat-item">
            <div class="stat-number">{len(df):,}</div>
            <div class="stat-label">商品总数</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{df['main_category'].nunique()}</div>
            <div class="stat-label">主要类别</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{df['store'].nunique():,}</div>
            <div class="stat-label">店铺数量</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${df['price'].mean():.2f}</div>
            <div class="stat-label">平均价格</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">{df['average_rating'].mean():.2f}</div>
            <div class="stat-label">平均评分</div>
        </div>
    </div>

    <h2>1. 数据概览</h2>
    <p>商品价格集中在 $28-$40 区间，评分普遍较高（平均 4.14），评论数分布极不均衡，少数商品占据大部分评论。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['price_dist']}" alt="价格分布"></div>
    <div class="img-container"><img src="data:image/png;base64,{images['rating_dist']}" alt="评分分布"></div>
    <div class="img-container"><img src="data:image/png;base64,{images['review_dist']}" alt="评论数分布"></div>

    <h2>2. 类别分析</h2>
    <p>商品数量最多的前10个类别中，<strong>{category_stats.index[0]}</strong> 类商品数量最多，平均评分均在 4.0 以上，其中 <strong>{category_stats['avg_rating'].idxmax()}</strong> 评分最高。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['category']}" alt="类别分析"></div>

    <h2>3. 店铺分析</h2>
    <p>头部店铺集中了大部分评论量，前10店铺的总评论数占整体比例较高，但店铺平均评分差异不大。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['store']}" alt="店铺分析"></div>

    <h2>4. 价格与评分关系</h2>
    <p>中等价位商品（$20-$60）数量最多，评分稳定在 4.1 左右；低价位商品评分略低，高价位商品评分波动较大但样本少。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['price_rating']}" alt="价格评分关系"></div>

    <h2>5. 商品特征词云</h2>
    <p>从商品特点中提取的高频词显示，消费者关注的材质（如 leather, cotton）、款式（如 casual, sneaker）等是核心卖点。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['wordcloud']}" alt="特征词云"></div>

    <h2>6. 视频对商品表现的影响</h2>
    <p>有视频的商品在平均评论数、平均评分和平均价格上均高于无视频商品，说明视频内容有助于提升商品热度和用户信任。</p>
    <div class="img-container"><img src="data:image/png;base64,{images['video']}" alt="视频影响"></div>

    <h2>7. 核心结论与建议</h2>
    <ul>
        <li><strong>定价策略：</strong>主流商品价格集中在 $28-$40，新商品可参考此区间，同时关注高性价比（低价高评分）机会。</li>
        <li><strong>内容优化：</strong>视频商品表现更优，建议卖家增加商品视频，尤其是高潜力新品。</li>
        <li><strong>选品方向：</strong>关注高频特征词（如 leather, casual, comfort），结合评分高的类别（如 {category_stats['avg_rating'].idxmax()}）开发新品。</li>
        <li><strong>店铺运营：</strong>头部店铺评论集中，新卖家需通过差异化定位或优质内容突围。</li>
    </ul>

    <div class="footer">
        报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 数据分析师：XXX
    </div>
</div>
</body>
</html>
"""

# 保存 HTML 文件
report_path = '电商数据分析报告.html'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"报告已生成：{os.path.abspath(report_path)}")