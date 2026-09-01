import joblib
import pandas as pd

rf = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

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
    return "Fake" if prediction[0] == 1 else "Real"

if __name__ == "__main__":
    # Example: an account with very few followers, following many, no profile pic,
    # no bio, no external link, private, username full of digits, name==username --
    # classic fake-account shape
    result = predict_fake_account(
        followers=50, follows=3000, profile_pic=0, description_length=0,
        external_url=0, private=1, posts=1,
        username_digit_ratio=0.4, fullname_digit_ratio=0.3,
        fullname_word_count=1, name_equals_username=1
    )
    print("Prediction:", result)