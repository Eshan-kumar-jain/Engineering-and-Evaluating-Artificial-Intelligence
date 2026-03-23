from model.randomforest import RandomForest



def model_predict(data, df, name):
    if data.X_train is None:
        print("Skipping — not enough data for this level.")
        return
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)



def model_evaluate(model, data):
    model.print_results(data)