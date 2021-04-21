from flask import Flask, jsonify, request, render_template
import time

if __name__ == '__main__':

    app = Flask(__name__)

    data = {}

    tony = False
    # while True:
    #     print("This is {}".format(tony))
    #     time.sleep(1)

    @app.route('/')
    def home(tony=None):
        print("hi")
        tony = not tony
        return jsonify({"hi":"tony"})

    @app.route('/update', methods=['POST'])
    def create_update():
        request_data = request.get_json()
        new_store = {
            'name': request_data['name'],
            'items': []
        }
        return jsonify(new_store)
        # pass


    # get /store/<name> data: {name :}
    @app.route('/store/<string:name>')
    def get_store(name):
        for store in stores:
            if store['name'] == name:
                return jsonify(store)
        return jsonify({'message': 'store not found'})
        # pass


    # get /store
    @app.route('/store')
    def get_stores():
        return jsonify({'stores': stores})
        # pass


    # post /store/<name> data: {name :}
    @app.route('/store/<string:name>/item', methods=['POST'])
    def create_item_in_store(name):
        request_data = request.get_json()
        for store in stores:
            if store['name'] == name:
                new_item = {
                    'name': request_data['name'],
                    'price': request_data['price']
                }
                store['items'].append(new_item)
                return jsonify(new_item)
        return jsonify({'message': 'store not found'})
        # pass


    # get /store/<name>/item data: {name :}
    @app.route('/store/<string:name>/item')
    def get_item_in_store(name):
        for store in stores:
            if store['name'] == name:
                return jsonify({'items': store['items']})
        return jsonify({'message': 'store not found'})

        # pass


    app.run(port=5051)
    while True:
        print("This is {}".format(tony))
        time.sleep(1)
