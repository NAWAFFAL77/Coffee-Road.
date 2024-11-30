from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
from bidi.algorithm import get_display
import arabic_reshaper
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # استخدم Backend غير تفاعلي
import matplotlib.pyplot as plt
import io
import base64
import pickle
import numpy as np
from flask_ngrok import run_with_ngrok

# Load the saved model and TF-IDF
try:
    with open("lr_model.pkl", "rb") as model_file:
        lr_model = pickle.load(model_file)

    with open("tfidf.pkl", "rb") as tfidf_file:
        tfidf = pickle.load(tfidf_file)

    print("Model and TF-IDF loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'lr_model.pkl' and 'tfidf.pkl' exist in the directory.")
    raise

# Define categories
categories = {
    "study": ["هادئ", "مريح", "إنترنت", "تركيز", "دراسة", "مناسب للدراسة", "مريح للدراسة"],
    "family": ["عائلات", "مناسب للعائلات", "مريح", "آمن", "مناسب للأطفال", "مريح للعائلات"],
    "price": ["رخيص", "أسعار معقولة", "سعر مناسب", "مناسب للميزانية", "قيمة مقابل المال"],
    "service": ["خدمة ممتازة", "موظفين محترمين", "خدمة جيدة", "التعامل الجيد", "مساعدة سريعة", "دعم ممتاز"]
}




def analyze_categories(reviews):
    scores = {category: 0 for category in categories}
    for category, keywords in categories.items():
        for keyword in keywords:
            scores[category] += reviews.count(keyword)
    return scores


def predict_coffee_shop_rating(coffee_shop_name, data):
    shop_comments = data[data['Coffee_Shop'].str.contains(coffee_shop_name, case=False, na=False)]
    if shop_comments.empty:
        return None, None, None
    shop_comments['Review'] = shop_comments['Review'].fillna('')

    shop_features = tfidf.transform(shop_comments['Review']).toarray()
    predicted_labels = lr_model.predict(shop_features)

    positive_count = np.sum(predicted_labels == 1)
    neutral_count = np.sum(predicted_labels == 2)
    negative_count = np.sum(predicted_labels == 0)
    avg_predicted_rating = (positive_count * 4.5 + neutral_count * 3 + negative_count * 2) / len(predicted_labels)

    return positive_count, neutral_count, negative_count, round(avg_predicted_rating, 1)


def generate_wordcloud(text):
    if not text.strip():
        return None
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    wordcloud = WordCloud(font_path="fonts/NotoSansArabic-VariableFont_wdth,wght.ttf",
                          background_color="white").generate(bidi_text)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import io
import base64

def generate_pie_chart(positive, neutral, negative):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive, neutral, negative]
    colors = ['#4caf50', '#2196f3', '#f44336']

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,textprops={'color': 'white'})
    plt.axis('equal')

    # تغيير خلفية الرسم البياني إلى الرمادي
    fig = plt.gcf()  # الحصول على الشكل الحالي
    fig.patch.set_facecolor('#333')  # تحديد الخلفية الرمادية

    img = io.BytesIO()
    plt.savefig(img, format='PNG')  # حفظ الصورة
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')



app = Flask(__name__)
app.secret_key = "a29c8f7d41e9e2344f1b21b8d172a6c"

DATA_PATH = "/Users/nawafall/PycharmProjects/pythonProject3/Corrected_Review_Data.xlsx"

run_with_ngrok(app)  # هذا يربط التطبيق بـ ngrok

@app.route('/some_route', methods=['GET', 'POST'])
def some_page():
    return render_template('some_template.html', back_url=request.referrer)

# بيانات النقاط للخريطة
locations = [
    {"name": "Riyadh", "lat": 24.7136, "lng": 46.6753},
    {"name": "Jeddah", "lat": 21.4858, "lng": 39.1925},
    {"name": "Dammam", "lat": 26.3927, "lng": 49.9777}
]

@app.before_request
def save_previous_page():
    # حفظ الرابط السابق في الجلسة
    if request.referrer:
        session['previous_page'] = request.referrer

# Route for map page
@app.route('/map')
def map_page():
    return render_template('map.html')


# API endpoint for locations
@app.route('/locations')
def get_locations():
    return jsonify(locations)

try:
    data = pd.read_excel(DATA_PATH)
    data.columns = data.columns.str.strip()
    print("تم تحميل بيانات الملف بنجاح.")
except Exception as e:
    data = None
    print(f"خطأ في تحميل البيانات: {e}")

users = {"admin": "password", "user1": "pass123"}


@app.route('/')
def home():
    return redirect(url_for('login'))

# باقي الكود كما هو...


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('select_city'))
        return render_template('login.html', error="Invalid email or password. Please try again.")
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="كلمتا المرور غير متطابقتين.")
        if username in users:
            return render_template('signup.html', error="اسم المستخدم موجود بالفعل.")

        users[username] = password
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/select_city', methods=['GET', 'POST'])
def select_city():
    if 'username' not in session:
        return redirect(url_for('login'))
    if data is None:
        return "خطأ في تحميل البيانات. يرجى التأكد من صحة ملف البيانات."

    cities = data['city'].dropna().unique().tolist()

    if request.method == 'POST':
        selected_city = request.form.get('city')
        session['selected_city'] = selected_city
        return redirect(url_for('select_neighborhood'))

    return render_template('city.html', cities=cities)


@app.route('/select_neighborhood', methods=['GET', 'POST'])
def select_neighborhood():
    if 'username' not in session or 'selected_city' not in session:
        return redirect(url_for('login'))
    selected_city = session['selected_city']
    neighborhoods = data[data['city'] == selected_city]['neighborhood'].dropna().unique().tolist()
    if request.method == 'POST':
        selected_neighborhood = request.form.get('neighborhood')
        session['selected_neighborhood'] = selected_neighborhood
        return redirect(url_for('select_coffee_shop'))
    return render_template('neighborhood.html', neighborhoods=neighborhoods, city=selected_city)


@app.route('/select_coffee_shop', methods=['GET', 'POST'])
def select_coffee_shop():
    if 'username' not in session or 'selected_neighborhood' not in session:
        return redirect(url_for('login'))

    # حفظ الرابط السابق في الجلسة
    session['previous_page'] = request.referrer

    selected_neighborhood = session['selected_neighborhood']
    coffee_shops = data[data['neighborhood'] == selected_neighborhood]['Coffee_Shop'].dropna().unique().tolist()

    if request.method == 'POST':
        selected_shop = request.form.get('shop')
        session['selected_shop'] = selected_shop
        return redirect(url_for('dashboard'))

    return render_template('coffee_shops.html', shops=coffee_shops, neighborhood=selected_neighborhood)


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # تحقق من أن المستخدم قد سجل الدخول واختار مقهى
    if 'username' not in session or 'selected_shop' not in session:
        return redirect(url_for('login'))

    # حفظ الرابط السابق في الجلسة للتمكن من الرجوع إليه
    session['previous_page'] = request.referrer

    selected_shop = session['selected_shop']
    shop_data = data[data['Coffee_Shop'] == selected_shop]

    # استخراج المراجعات الإيجابية والسلبية للمقهى
    positive_reviews = " ".join(shop_data[shop_data['Sentiment'] == 'Positive']['Review'].dropna())
    negative_reviews = " ".join(shop_data[shop_data['Sentiment'] == 'Negative']['Review'].dropna())

    # استدعاء الدالة لحساب الإحصائيات
    positive_count, neutral_count, negative_count, avg_rating = predict_coffee_shop_rating(selected_shop, data)
    total_comments = positive_count + neutral_count + negative_count  # حساب إجمالي التعليقات
    category_scores = analyze_categories(positive_reviews + negative_reviews)

    # حساب التقييمات من جوجل
    google_rating = shop_data['stars'].mean()

    # توليد كلمة سحابية للمراجعات الإيجابية والسلبية
    positive_wordcloud = generate_wordcloud(positive_reviews)
    negative_wordcloud = generate_wordcloud(negative_reviews)

    # توليد مخطط دائري (Pie Chart) للتصنيف
    pie_chart = generate_pie_chart(positive_count, neutral_count, negative_count)

    # عرض صفحة الـ Dashboard مع البيانات اللازمة
    return render_template(
        'dashboard.html',
        shop=selected_shop,
        positive_count=positive_count,
        neutral_count=neutral_count,
        negative_count=negative_count,
        avg_rating=avg_rating,
        google_rating=round(google_rating, 1) if google_rating else "N/A",
        category_scores=category_scores,
        total_comments=total_comments,
        positive_wordcloud=positive_wordcloud,
        negative_wordcloud=negative_wordcloud,
        pie_chart=pie_chart
    )


@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if 'username' not in session or 'selected_city' not in session:
        return redirect(url_for('login'))

    selected_city = session['selected_city']
    neighborhoods = data[data['city'] == selected_city]['neighborhood'].dropna().unique().tolist()
    categories = {
        "study": ["هادئ", "مريح", "إنترنت", "تركيز", "دراسة", "مناسب للدراسة", "مريح للدراسة"],
        "family": ["عائلات", "مناسب للعائلات", "مريح", "آمن", "مناسب للأطفال", "مريح للعائلات"],
        "price": ["رخيص", "أسعار معقولة", "سعر مناسب", "مناسب للميزانية", "قيمة مقابل المال"],
        "service": ["خدمة ممتازة", "موظفين محترمين", "خدمة جيدة", "التعامل الجيد", "مساعدة سريعة", "دعم ممتاز"]
    }  # Define categories with corresponding keywords

    # حفظ الصفحة السابقة في session
    session['previous_page'] = request.referrer

    if request.method == 'POST':
        category = request.form.get('category')
        neighborhood = request.form.get('neighborhood', None)

        if neighborhood:
            filtered_data = data[(data['city'] == selected_city) & (data['neighborhood'] == neighborhood)]
        else:
            filtered_data = data[data['city'] == selected_city]

        recommendations = []

        for shop in filtered_data['Coffee_Shop'].unique():
            shop_data = filtered_data[filtered_data['Coffee_Shop'] == shop]
            reviews = " ".join(shop_data['Review'].dropna())
            scores = analyze_categories(reviews)

            if scores[category] > 0:
                google_rating = shop_data['stars'].mean()  # Use Google Maps rating
                total_comments = len(shop_data)  # Count total comments for the shop

                # Add the neighborhood information here
                neighborhood_name = shop_data['neighborhood'].iloc[0]  # Assuming neighborhood is the same for the shop

                # Display only the selected category score
                category_score = {category: scores[category]}  # Display only the selected category score

                recommendations.append({
                    'Coffee_Shop': shop,
                    'category_scores': category_score,  # Show only selected category score
                    'total_comments': total_comments,
                    'google_rating': round(google_rating, 1) if not pd.isna(google_rating) else "N/A",
                    'neighborhood': neighborhood_name  # Add neighborhood here
                })

        # Sort recommendations based on the selected category score
        recommendations = sorted(recommendations, key=lambda x: x['category_scores'][category], reverse=True)

        return render_template('recommendation_results.html', recommendations=recommendations, category=category, categories=categories)  # Pass category and categories to the template

    return render_template('recommendation_form.html', neighborhoods=neighborhoods, categories=categories)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/back')
def back():
    # استخدم الصفحة السابقة من الجلسة أو عد إلى الصفحة الرئيسية
    previous_page = session.get('previous_page', 'home')
    return redirect(url_for(previous_page))


# Ensure the app runs on the correct port
if __name__ == '__main__':
    app.run()  # إزالة port