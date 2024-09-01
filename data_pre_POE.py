from concurrent.futures import ThreadPoolExecutor
import json
import copy
from tqdm import tqdm
import queue
import os

class Get_Data():
    def __init__(self, source_path, save_path):
        self.data_template = {
            "conversation": [
                {
                    "system": "",
                    "input": "xxx",
                    "output": "xxx"
                }
            ]
        }
        self.source_path = source_path
        self.save_path = save_path
    
    def build_data(self, chat_rsp):
        temp = copy.deepcopy(self.data_template)
        temp['conversation'][0]['input'] = ''
        temp['conversation'][0]['output'] = chat_rsp
        return temp
    
    def run(self):
        read_queue = queue.Queue(maxsize=100)
        process_queue = queue.Queue(maxsize=100)

        with ThreadPoolExecutor(max_workers=3) as pool:
            reader_future = pool.submit(self.reader_worker, read_queue)
            processor_future = pool.submit(self.processor_worker, read_queue, process_queue)
            saver_future = pool.submit(self.saver_worker, process_queue)

            reader_future.result()
            processor_future.result()
            saver_future.result()

    def save(self, item):
        with open(self.save_path, 'a', encoding='utf-8') as f:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")
    
    def reader_worker(self, read_queue):
        for chunk in self.read_jsonl_in_chunks(chunk_size=100):
            read_queue.put(chunk)
        read_queue.put(None)  # Signal that reading is complete

    def processor_worker(self, read_queue, process_queue):
        while True:
            chunk = read_queue.get()
            if chunk is None:
                process_queue.put(None)
                break
            for item in chunk:
                processed_item = self.build_data(item)
                process_queue.put(processed_item)

    def saver_worker(self, process_queue):
        while True:
            item = process_queue.get()
            if item is None:
                break
            self.save(item)
        
    def read_jsonl_in_chunks(self, chunk_size=100):
        chunk = []
        with open(self.source_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Reading file"):
                chunk.append(json.loads(line.strip())['content'])
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

def get_file_paths(directory):
    file_paths = []
    file_names = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            file_names.append(file)
    
    return file_paths, file_names

if __name__ == '__main__':
    source_path = '/group_share/datasets/wanjuan/OpenDataLab___WanJuanCC/raw/jsonl'
    file_paths, file_names = get_file_paths(source_path)
    save_path = '/group_share/datasets/wanjuan_xtuner'
    os.makedirs(save_path, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=25) as pool:
        futures = []
        for file_path, file_name in zip(file_paths, file_names):
            save_file_path = os.path.join(save_path, file_name)
            gda = Get_Data(file_path, save_file_path)
            futures.append(pool.submit(gda.run))
        
        for future in futures:
            future.result()
    
    print('done')