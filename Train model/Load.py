model_path = 'model/xgboost-model.json'
model = xgb.Booster()
model.load_model(model_path)
