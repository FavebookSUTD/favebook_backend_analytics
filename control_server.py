#! /usr/bin/python3

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from spark_controller import ClusterController, JobController, JobRunningException

app = Flask(__name__)
CORS(app)

with open('/etc/opt/control-server/mysql-ip') as file:
    mysql_ip = file.read().strip()
with open('/etc/opt/control-server/mongo-ip') as file:
    mongo_ip = file.read().strip()

cluster_ctrl = ClusterController()
job_ctrl = JobController(cluster_ctrl, mysql_ip, mongo_ip)

@app.route('/analytics/start', methods=['POST'])
def start():
    # get args
    job = request.json['job']
    assert job in JobController.JOB_SCRIPTS
    nodes = int(request.json['nodes'])
    assert nodes in range(1, 9)
    # start job
    try:
        job_ctrl.start_job(job, nodes)
        return '', 201
    except JobRunningException:
        return '', 409

@app.route('/analytics/status')
def status():
    job = request.args['job']
    assert job in JobController.JOB_SCRIPTS
    if any(t.name == job for t in job_ctrl.job_queue):
        status = 'running'
    elif job_ctrl.job_success[job]:
        status = 'success'
    else:
        status = 'fail'
    return jsonify({
        "data": {
            "job": job,
            "status": status,
        }
    })

@app.route('/analytics/cluster_health')
def cluster_health():
    return jsonify({
        'data': cluster_ctrl.get_cluster_health()
    })

@app.route('/analytics/pearson_result')
def pearson_result():
    if job_ctrl.pearson_result == None:
        return '', 400
    return jsonify({
        'data': job_ctrl.pearson_result
    })

@app.route('/analytics/tfidf_result')
def tfidf_result():
    if not job_ctrl.job_success['tfidf']:
        return '', 400
    page_num = int(request.args['pageNum'])
    page_size = int(request.args['pageSize'])
    filter_ = request.args.get('filter')
    assert filter_ in ('reviewId', 'reviewText', 'featureLength', None)
    review_text = request.args.get('reviewText')
    review_id = request.args.get('reviewId')
    feature_length = int(request.args.get('featureLength') or 0)

    if filter_ == 'reviewId':
        if not review_id: return '', 404
        filter_func = lambda obj: obj['id'] == review_id
    elif filter_ == 'reviewText':
        if not review_text: return '', 404
        filter_func = lambda obj: review_text in obj['reviewText']
    elif filter_ == 'featureLength':
        if not feature_length: return '', 404
        assert feature_length
        filter_func = lambda obj: obj['feature_indices_size'] == feature_length
    else:   
        filter_func = lambda obj: True

    with open('/tmp/reviews_tfidf.json') as file:
        for _ in range((page_num-1)*page_size):
            file.readline()
        records = []
        for _ in range(page_size):
            records.append(json.dumps(file.readline()))
        count = page_num * page_size
        for _ in file:
            count += 1

    reviews = [
        {
            'reviewId': obj['id'],
            'reviewText': obj['reviewText'],
            'featureLength': obj['feature_indices_size'],
            'featureIndices': obj['feature_indices'],
            'featureWeights': obj['feature_values'],
            'sentiment': "positive" if float(obj['overall']) >= 3.5 else "negative" if float(obj['overall']) <= 2.5 else 'neutral'
        }
        for obj in records
        if filter_func(obj)
    ]

    return jsonify({
        'data': {
            'reviews': reviews,
            'totalCount': count
        }
    })

@app.route('/datanode_register', methods=['POST'])
def datanode_register():
    cluster_ctrl.register_datanode(request.remote_addr)
    return ''
