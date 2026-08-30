from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))

def predict_fake_account(followers, follows, profile_pic, description_length,
                          external_url, private, posts):
    ratio = followers / (follows + 1)
    completeness = (
        int(profile_pic) +
        int(description_length > 0) +
        int(external_url) +
        int(1 - private)
    ) / 4
    posts_per_follower = posts / (followers + 1)

    input_data = scaler.transform([[ratio, completeness, posts_per_follower]])
    prediction = rf.predict(input_data)
    return "Fake" if prediction[0] == 1 else "Real"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        followers = int(request.form['followers'])
        follows = int(request.form['follows'])
        profile_pic = int(request.form['profile_pic'])
        description_length = int(request.form['description_length'])
        external_url = int(request.form['external_url'])
        private = int(request.form['private'])
        posts = int(request.form['posts'])

        result = predict_fake_account(
            followers, follows, profile_pic, description_length,
            external_url, private, posts
        )

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
