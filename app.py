from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import os
import pandas as pd
from MLModel import MLModel

app = Flask(__name__)
api = Api(app, version='1.0', title='API Documentation')

obj_mlmodel = MLModel()

predict_model = api.model('PredictModel', {
    'inference_row': fields.List(fields.Raw, required=True,
                                 description='A row of data for inference')
})

file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

ns = api.namespace('model', description='Model operations')


@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self):
        args = file_upload.parse_args()
        uploaded_file = args['file']
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400

        data_path = './Original Data/Clean_Dataset.csv'
        uploaded_file.save(data_path)

        try:
            df = pd.read_csv(data_path)
            df = obj_mlmodel.preprocessing_pipeline(df)
            print(df.head())
            mape_score, model = obj_mlmodel.train_and_save_model(df)
            obj_mlmodel.save_model(model, 'artifacts/models/model.pkl')
            df.to_csv('artifacts/preprocessed_data/saved_dataframe_new.csv', index=False)
            os.remove(data_path)

            return {'message': 'Model Trained Successfully',
                    'mape_score': mape_score}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500


@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self):
        try:
            data = request.get_json()
            if 'inference_row' not in data:
                return {'error': 'No inference_row found'}, 400
            print('checkpoint in app.py')
            infer_array = data['inference_row']
            print('checkpoint 3')
            df = obj_mlmodel.preprocessing_pipeline_inference(infer_array)
            y_pred = obj_mlmodel.predict(df)
            return {'message': 'Inference Successful', 'prediction': str(y_pred)}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
