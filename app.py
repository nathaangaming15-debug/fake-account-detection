from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

FEATURES = ['follower_following_ratio', 'completeness_score', 'posts_per_follower',
            'username_digit_ratio', 'fullname_digit_ratio', 'fullname_word_count',
            'name_equals_username']

def predict_fake_account(followers, follows, profile_pic, description_length,
                          external_url, private, posts,
                          username_digit_ratio, fullname_digit_ratio,
                          fullname_word_count, name_equals_username):
    ratio = followers / (follows + 1)
    completeness = (
        int(profile_pic) +
        int(description_length > 0) +
        int(external_url) +
        int(1 - private)
    ) / 4
    posts_per_follower = posts / (followers + 1)

    input_df = pd.DataFrame([[
        ratio, completeness, posts_per_follower,
        username_digit_ratio, fullname_digit_ratio,
        fullname_word_count, name_equals_username
    ]], columns=FEATURES)

    input_data = scaler.transform(input_df)
    prediction = rf.predict(input_data)
    proba = rf.predict_proba(input_data)[0][1]
    return ("Fake" if prediction[0] == 1 else "Real"), round(proba * 100, 1)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    if request.method == 'POST':
        followers = int(request.form['followers'])
        follows = int(request.form['follows'])
        profile_pic = int(request.form['profile_pic'])
        description_length = int(request.form['description_length'])
        external_url = int(request.form['external_url'])
        private = int(request.form['private'])
        posts = int(request.form['posts'])
        username_digit_ratio = float(request.form['username_digit_ratio'])
        fullname_digit_ratio = float(request.form['fullname_digit_ratio'])
        fullname_word_count = int(request.form['fullname_word_count'])
        name_equals_username = int(request.form['name_equals_username'])

        result, confidence = predict_fake_account(
            followers, follows, profile_pic, description_length,
            external_url, private, posts,
            username_digit_ratio, fullname_digit_ratio,
            fullname_word_count, name_equals_username
        )

    return render_template('index.html', result=result, confidence=confidence)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)