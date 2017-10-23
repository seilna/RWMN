import Queue
import threading
import h5py
import numpy as np
from time import time
from IPython import embed
from hickle import load
from tensorflow import flags
FLAGS = flags.FLAGS

class LTM_Queue(object):
    def __init__(self, story_filename, qa_filelist, capacity, batch_size, num_threads):
        self.qa_data = {}
        for qa_file in qa_filelist:
            imdb_key = qa_file.split('/')[-1].split('.h5')[0]
            self.qa_data[imdb_key] = h5py.File(qa_file)

        if FLAGS.video_features == True or FLAGS.sub_with_video_features == True:
            self.story = load(story_filename)
        else: self.story = h5py.File(story_filename)
        self.queue = Queue.Queue(capacity)
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_movie = len(self.qa_data)

    def sequential_iterator(self):
        # For full evaluation, check around all validation QA example
        for movie_index in range(self.num_movie):
            imdb_key = self.qa_data.keys()[movie_index]
            num_examples_per_movie = len(self.qa_data[imdb_key]['qid'])
            mini_batch = {}
            qa_indices = range(num_examples_per_movie)
            for key in self.qa_data[imdb_key].keys():
                if ("video" in key) or ("subtt" in key): continue
                mini_batch[key] = self.qa_data[imdb_key][key][qa_indices]
            if (FLAGS.video_features == True) or (FLAGS.sub_with_video_features == True):
                mini_batch["rgb"] = self.story[imdb_key]["rgb"]
                mini_batch["sub"] = self.story[imdb_key]["sub"]
            else: 
                mini_batch["story"] = self.story[imdb_key]

            
            yield mini_batch


    def iterator(self):
        while True:
            # Randomly select 1 movie
            mini_batch = {}
            movie_index = np.random.choice(self.num_movie, 1)
            movie_index = int(movie_index)
            imdb_key = self.qa_data.keys()[movie_index]
            if FLAGS.video_features == True or FLAGS.sub_with_video_features == True:
                mini_batch['rgb'] = self.story[imdb_key]['rgb']
                mini_batch['sub'] = self.story[imdb_key]['sub']
            else:
                mini_batch['story'] = self.story[imdb_key]

            # and randomly select QA examples * mini-batch size
            num_examples = len(self.qa_data[imdb_key]['query'])
            batch_size = min([self.batch_size, num_examples])
            qa_indices = np.random.choice(num_examples, batch_size, replace=False)
            qa_indices = list(qa_indices)
            qa_indices.sort()

            # generate mini-batch examples
            for key in self.qa_data[imdb_key].keys():
                # For saving time, ignore movie/subtt representation
                if ('video' in key) or ('subtt' in key): continue
                mini_batch[key] = self.qa_data[imdb_key][key][qa_indices]
            yield mini_batch

    def get_inputs(self):
        batch_data = self.queue.get(block=True)
        return batch_data

    def thread_main(self):
        # Getting batch data using iterator for h5py file and
        # push them in the queue
        for mini_batch in self.iterator():
            self.queue.put(mini_batch, block=True)

    def thread_main_sequential(self):
        for example in self.sequential_iterator():
            self.queue.put(example, block=True)

    def start_threads(self, sequential=False):
        if sequential == False:
            for _ in range(self.num_threads):
                self.thread = threading.Thread(target=self.thread_main, args=())
                self.thread.daemon = True
                self.thread.start()
        elif sequential == True:
            for _ in range(self.num_threads):
                self.thread = threading.Thread(target=self.thread_main_sequential, args=())
                self.thread.deamon = True
                self.thread.start()


class BatchQueue(object):
    def __init__(self, filename, capacity, batch_size, num_threads):
        self.data = h5py.File(filename)
        self.num_examples = len(self.data['video_rep'])
        self.queue = Queue.Queue(capacity)
        self.batch_size = batch_size
        self.num_threads = num_threads

    def memnet_iterator(self):
        while True:
            # Randomly sample the data indices (* batch_size)
            indices = np.random.choice(self.num_examples, self.batch_size, replace=False)
            indices.sort()
            indices = list(indices)

            # Yield mini-batch data using random indices
            # (but, indices have constraints which is always increasing-order... fix this!)
            mini_batch = {}
            for key in self.data.keys():
                # For saving time, ignore movie/subtt representation
                if ('video' in key) or ('subtt' in key): continue
                mini_batch[key] = self.data[key][indices]
            yield mini_batch

    def ltm_iterator(self):
        pass

    def get_inputs(self):
        # Getting batched data
        batch_data = self.queue.get(block=True)
        return batch_data

    def thread_main(self):
        # Getting batch data using iterator for h5py file and
        # push them in the queue
        for mini_batch in self.memnet_iterator():
            self.queue.put(mini_batch, block=True)

    def start_threads(self):
        for _ in range(self.num_threads):
            self.thread = threading.Thread(target=self.thread_main, args=())
            self.thread.daemon = True
            self.thread.start()
