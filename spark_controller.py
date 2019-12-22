#! /usr/bin/python3

from subprocess import run, Popen, PIPE, CalledProcessError
from threading import Thread
from urllib.request import urlopen
from time import sleep
import csv
import json
import socket
import fabric

class JobRunningException(Exception):
    pass

class ClusterController:

    KEY_FILE_NAME = '/etc/opt/control-server/my-hadoop-key.pem'

    def __init__(self):
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.datanodes = []
        self.state = None

    @staticmethod
    def ssh_node(ip):
        return fabric.Connection(
            host=ip,
            user='ubuntu',
            connect_kwargs={
                'key_filename': ClusterController.KEY_FILE_NAME
            }
        )

    def register_datanode(self, ip):
        if ip not in self.datanodes:
            self.datanodes.append(ip)

    def get_cluster_state(self):
        with urlopen(f"http://{self.local_ip}:8080/json") as page:
            self.state = json.load(page)
        return self.state

    def set_node_running(self, ip, running):
        print(f"{'Starting' if running else 'Stopping'} slave node at {ip}")
        with ClusterController.ssh_node(ip) as c:
            c.run(f'sudo systemctl {"start" if running else "stop"} spark-slave')
        print(f"{'Started' if running else 'Stopped'} slave node at {ip}")

    def wait_cluster_size(self, nodes):
        print(f"Waiting for Spark to report cluster size {nodes}")
        cluster_size = None
        while cluster_size != nodes:
            cluster_size = self.get_cluster_state()['aliveworkers']
            sleep(0.5)

    def scale_cluster(self, nodes):
        print(f"Scaling cluster size to {nodes}")
        scaling_threads = []
        for inst in self.datanodes[:nodes]:
            thread = Thread(target=self.set_node_running, args=(inst, True))
            thread.start()
            scaling_threads.append(thread)
        for inst in self.datanodes[nodes:]:
            thread = Thread(target=self.set_node_running, args=(inst, False))
            thread.start()
            scaling_threads.append(thread)
        for t in scaling_threads:
            t.join()
        self.wait_cluster_size(nodes)

    def _get_node_health(self, name, ip, cluster_state, return_):
        try:
            with ClusterController.ssh_node(ip) as c:
                mem = c.run('free -m', hide=True).stdout
                mem = int(mem.split('\n')[1].split()[1])
                cores = c.run('lscpu | grep "^CPU(s)"', hide=True).stdout
                cores = int(cores.split()[1])
            alive = any(w['host'] == ip and w['state'] == 'ALIVE' for w in cluster_state['workers'])
            status = 'alive' if alive else 'ready'
            return_.append({
                "name": name,
                "address": ip,
                "status": status,
                "cores": cores,
                "memoryMB": mem,
            })
        except:
            return_.append({
                "name": name,
                "address": ip,
                "status": 'failed',
                "cores": 0,
                "memoryMB": 0,
            })
    
    def get_cluster_health(self):
        threads = []
        cluster_state = self.get_cluster_state()
        nodes = []
        for i in range(8):
            name = 'datanode' + str(i+1)
            ip = self.datanodes[i]
            t = Thread(target=self._get_node_health, args=(name, ip, cluster_state, nodes))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return nodes

class JobController:

    JOB_SCRIPTS = {
        'pearson': '/opt/control-server/pearson.py',
        'tfidf': '/opt/control-server/basic_tfidf.py',
    }

    def __init__(self, cluster_controller, mysql_ip, mongo_ip):
        self.job_success = {
            'pearson': False,
            'tfidf': False,
        }
        self.job_queue = []
        self.cluster_controller = cluster_controller
        self.pearson_result = None
        self.mysql_ip = mysql_ip
        self.mongo_ip = mongo_ip
    
    def _spark_submit(self, job):
        try:
            proc = run(
                [
                    '/opt/spark/bin/spark-submit',
                    '--master', f"spark://{self.cluster_controller.local_ip}:7077",
                    '--name', job,
                    JobController.JOB_SCRIPTS[job],
                    self.cluster_controller.local_ip,
                ],
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                check=True,
                encoding='utf8',
            )
            self.job_success[job] = True
            print(f"{job} job finished successfully")
            return proc
        except CalledProcessError:
            print(f"Job thread for {job} failed")
            self.job_success[job] = False

    def _tfidf_job(self, nodes):
        print("Clearing old historical_reviews from HDFS.")
        run([
            '/opt/hadoop/bin/hdfs',
            'dfs',
            '-rm', '-r', '-f', 'hdfs:///user/root/historical_reviews',
        ])
        print("Loading historical_reviews from MySQL.")
        run(
            [
                '/opt/sqoop/bin/sqoop',
                'codegen',
                '--connect', f'jdbc:mysql://{self.mysql_ip}/50043db',
                '--username', 'faveadmin',
                '--password', 'password',
                '--table', 'historical_reviews',
                '--bindir', '/tmp',
            ],
            env={'HADOOP_HOME': '/opt/hadoop'},
            stdout=PIPE,
            stderr=PIPE,
        )
        run(
            [
                '/opt/sqoop/bin/sqoop',
                'import',
                '--connect', f'jdbc:mysql://{self.mysql_ip}/50043db',
                '--username', 'faveadmin',
                '--password', 'password',
                '--table', 'historical_reviews',
                '--bindir', '/tmp',
                '--columns', "id,asin,review_rating,review_text",
                '--enclosed-by', '"',
            ],
            env={'HADOOP_HOME': '/opt/hadoop'},
            stdout=PIPE,
            stderr=PIPE,
        )
        print("Clearing old reviews_tfidf.json from HDFS.")
        run([
            '/opt/hadoop/bin/hdfs',
            'dfs',
            '-rm', '-r', '-f', 'hdfs:///reviews_tfidf.json',
        ])
        self.cluster_controller.scale_cluster(nodes)
        print(f"Starting job tfidf")
        self._spark_submit('tfidf')
        self.cluster_controller.scale_cluster(0)
        run([
                '/opt/hadoop/bin/hdfs',
                'dfs',
                '-getmerge', 'hdfs:///reviews_tfidf.json', '/tmp/reviews_tfidf.json'
        ])

    def _pearson_job(self, nodes):
        print("Clearing old historical_reviews from HDFS.")
        run([
            '/opt/hadoop/bin/hdfs',
            'dfs',
            '-rm', '-r', '-f', 'hdfs:///user/root/historical_reviews',
        ])
        print("Loading historical_reviews from MySQL.")
        run(
            [
                '/opt/sqoop/bin/sqoop',
                'codegen',
                '--connect', f'jdbc:mysql://{self.mysql_ip}/50043db',
                '--username', 'faveadmin',
                '--password', 'password',
                '--table', 'historical_reviews',
                '--bindir', '/tmp',
                '--enclosed-by', '"',
            ],
            env={'HADOOP_HOME': '/opt/hadoop'},
            stdout=PIPE,
            stderr=PIPE,
        )
        run(
            [
                '/opt/sqoop/bin/sqoop',
                'import',
                '--connect', f'jdbc:mysql://{self.mysql_ip}/50043db',
                '--username', 'faveadmin',
                '--password', 'password',
                '--table', 'historical_reviews',
                '--bindir', '/tmp',
                '--columns', "id,asin,review_rating,review_text",
                '--enclosed-by', '"',
            ],
            env={'HADOOP_HOME': '/opt/hadoop'},
            stdout=PIPE,
            stderr=PIPE,    
        )
        print("Clearing old pearson_full.csv from HDFS.")
        run([
            '/opt/hadoop/bin/hdfs',
            'dfs',
            '-rm', '-r', '-f', 'hdfs:///pearson_full.csv',
        ])
        print("Clearing old meta.json from HDFS.")
        run([
            '/opt/hadoop/bin/hdfs',
            'dfs',
            '-rm', '-r', '-f', 'hdfs:///meta.json',
        ])
        print("Loading meta.json from MongoDB.")
        mongo = Popen(
            [
                'mongoexport',
                '--collection=kindle_metadata2',
                '--db=50043db',
                '--fields=asin,price',
                '--host=' + self.mongo_ip,
                '--port=27017',
                '--username=faveadmin',
                '--password=password',
                '--quiet',
            ],
            stdout=PIPE,
        )
        run(
            [
                '/opt/hadoop/bin/hdfs',
                'dfs',
                '-put', '-', 'hdfs:///meta.json',
            ],
            stdin = mongo.stdout,
        )
        self.cluster_controller.scale_cluster(nodes)
        print(f"Starting job pearson")
        proc = self._spark_submit('pearson')
        self.cluster_controller.scale_cluster(0)
        for line in proc.stdout.split('\n'):
            if 'pearson correlation = ' in line:
                correlation = float(line.split(' = ')[1])
                with open('/tmp/pearson_correlation', 'w') as file:
                    file.write(str(correlation))
        proc = run(
            [
                '/opt/hadoop/bin/hdfs',
                'dfs',
                '-cat', 'hdfs:///pearson_full.csv/*',
            ],
            stdout=PIPE,
            encoding='utf8',
        )
        points = list(csv.reader(proc.stdout.split('\n')))
        self.pearson_result = {
            'pearsonCoefficient': correlation,
            'reviewsAndPrices': [
                {'reviewLength': point[0], 'price': point[1]}
                for point in points
                if len(point) >= 2
            ],
            'totalCount': len(points),
        }

    def job_thread(self, job, nodes):
        print(f"Started job thread for {job}")
        if job == 'tfidf':
            self._tfidf_job(nodes)
        elif job == 'pearson':
            self._pearson_job(nodes)
        self.job_queue.pop(0)
        if len(self.job_queue) > 0:
            self.job_queue[0].start()
        print(f"Completed job thread for {job}")

    def start_job(self, job, nodes):
        if all(t.name != job for t in self.job_queue):
            thread = Thread(target=self.job_thread, name=job, args=(job, nodes))
            self.job_queue.append(thread)
            if len(self.job_queue) == 1:
                thread.start()
        else:
            print(f"Attempted to start {job} job, but it is already running.")
            raise JobRunningException()
