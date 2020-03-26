import pickle
import faiss
import numpy as np

class EmbedDatabase():

    def __init__(self,d=64):
       
        #d = 2048 #Size of embeddings
        self.index = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIDMap(self.index)
        print('FAISS index ready: ',self.index.is_trained)
        self.query_embeddings = []
        self.query_ids = []
        

    def addIndex(self,embeddings,ids):
        '''
        embeddings= np.shape(n,d)
        class_name, snip_id, trackid = np.shape(n,)
        '''

        #Adding to the main index to be searched
        self.index.add_with_ids(embeddings, ids)

        #Adding embedding to the query embeddings list
        self.query_embeddings.append(embeddings)
        self.query_ids.append(ids)


        return None

    def idAccuracy(self,k=6):
        '''
            query_embeddings = np.shape(n,d)
        '''

        qe = np.concatenate(self.query_embeddings,axis=0)
        print(qe.shape)
        ids = np.concatenate(self.query_ids,axis=0)

        n = qe.shape[0]

        D, I = self.index.search(qe, k)
        #Shape of I = np.shape(n,k)

        topk1,topk5 = self.topk(I,ids)

        return topk1,topk5
        

    def topk(self,output,target):
        '''
            output = np.size(n,k)
            target = np.size(n,)
        '''

        n = output.shape[0]
        topk1 = 0
        topk5 = 0

        for i in range(n):
            #The 0th match should be itself, look at the next one
            
            if target[i] == output[i][1]:
                topk1+=1
            if target[i] in output[i][1:]:
                topk5+=1

        topk1 = topk1/n
        topk5 = topk5/n
        
        return topk1,topk5
        
        



    
    '''def saveEmbeddings(self,type=None):
        if type='train':
            self.database_save_path = self.path_to_experiment+'embs_database_'+self.experiment_name+'.pkl'
            f = open(self.database_save_path,"wb")
            pickle.dump(self.embeddings,f)
            f.close()
        else:
            self.query_save_path = self.path_to_experiment+'embs_query'+self.experiment_name+'.pkl'
            f = open(self.query_save_path,"wb")
            pickle.dump(self.embeddings,f)
            f.close()

        return "Embeddings saved to disk"

    def embeddingAccuracy(self):

        database_embeddings = pickle.load(open(self.database_save_path, "rb" ))
        query_embeddings = pickle.load(open(self.query_save_path, "rb" ))

        for query_id in query_embeddings:
            distances = {}
            for db_id in database_embeddings:
                dist = torch.dist(query_embeddings[query_id][0],database_embeddings[db_id][0],2)
                distances[db_id].append

        return None'''


        

    
