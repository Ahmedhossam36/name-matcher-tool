if __name__ == '__main__':
    import joblib
    loaded_model = joblib.load('name_matching_model.pkl')
    loaded_vectorizer = joblib.load('vectorizer.pkl')
    sample_names = ['Ahmed Hossam mahrous ahmed']
    sample_vectorized = loaded_vectorizer.transform(sample_names)
    prediction = loaded_model.predict(sample_vectorized)
    print(prediction)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
