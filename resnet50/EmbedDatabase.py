import pickle
import faiss

class EmbedDatabase():

    def __init__(self,path_to_experiment,experiment_name,unique_keys,d=64):
        self.path_to_experiment = path_to_experiment
        self.experiment_name = experiment_name
        self.database_save_path, self.query_save_path = '',''
        #d = 2048 #Size of embeddings
        self.index = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIDMap(self.index)
        print('FAISS index ready: ',self.index.is_trained)

        self.unique_keys = unique_keys

    def getIndex(self,cat_code,snip_id,trackid):

        uk = pd.DataFrame(self.unique_keys)
        index_df = uk[(uk.cat_code==cat_code) & (uk.snip_id==snip_id) & (uk.trackid == trackid)].index.to_numpy()

        return int(index_df)
        

    def addIndex(self,embeddings,cat_code,snip_id,trackid):
        '''
        embeddings= np.shape(n,d)
        class_name, snip_id, trackid = np.shape(n,)
        '''

        #Get index value of class_name, snip_id, trackid
        n = embeddings.shape[0]
        ids = np.empty([n,])

        for i in range(n):
            index = self.getIndex(cat_code[i],snip_id[i],trackid[i])
            ids[i] = index
        
        self.index.add_with_ids(embeddings, ids)

        return None

    def accuracy(self,query_embeddings,cat_code,snip_id,trackid,k=1):
        '''
            query_embeddings = np.shape(n,d)
        '''

        n = query_embeddings.shape[0]
        ids = np.empty([n,])

        for i in range(n):
            index = self.getIndex(cat_code[i],snip_id[i],trackid[i])
            ids[i] = index

        D, I = index2.search(query_embeddings, k)
        #Shape of I = np.shape(n,k)

        topk1,topk5 = topk(I,ids)

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
            if target[i] == output[i][0]:
                topk1+=1
            if target[i] in output[i]:
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


        

    
