import os
import glob
import pickle
import subprocess
import pandas as pd

path = os.getcwd() + '/'

def extract_relations():
    
    print('Relationship Extraction\n')

    for f in glob.glob(path + "data/output/kg/*.txt"):        
        print("\nExtracting relations for " + f.split("/")[-1])
        os.chdir(path + '/stanford-openie')
        p = subprocess.Popen(['./process_large_corpus.sh',f,f + '-out.csv'], stdout=subprocess.PIPE)
        output, err = p.communicate()
        print('Input --> ',f)
        print('Output -->', f + '-out.csv')
    print('\n--------------------------------------------------------------------------------\n')
    


def build_knowledge_graph():
    # Create a list of pickle file names
    print('Postprocessing and building knowledge graph\n')
    pickles = []
    for file in glob.glob(path + "data/output/ner/*.pickle"):
        pickles.append(file)

    # Load each pickle file and create the resultant csv file
    for file in pickles:
        with open(file,'rb') as f:
            entities = pickle.load(f)

        # Add all the names in entity set
        entity_set = set(entities.keys())
        final_list = []
        file_name_list = file.split('/')[-1].split('.')[0].split('_')[2:]
        file_name = file_name_list[0]
        flag = True
        for str in file_name_list[1:]:
            file_name += '_'
            file_name += str
            print(file_name)

        df = pd.read_csv(path +"data/output/kg/"+file_name+".txt-out.csv", header=None)
        # Parse every row present in the intermediate csv file
        
        triplet = set()
        
        for i, j in df.iterrows():
            
            j[0] = j[0].strip()
            j[2] = j[2].strip()
            
            # If entity is present in entity set, only then parse futrther
            if j[0] in entity_set: #or j[2] in entity_set:
                added = False
                e2_sentence = j[2].split(' ')
                # Check every word in entity2, and add a new row triplet if it is present in entity2
                for entity in e2_sentence:
                    if entity in entity_set:
                        _ = (entities[j[0]], j[0], j[1], entities[entity], j[2])
                        triplet.add(_)
                        added = True
                if not added:
                    _ = (entities[j[0]], j[0] ,j[1] ,'O', j[2])
                    triplet.add(_)
            
            if j[2] in entity_set:
                added = False
                e2_sentence = j[0].split(' ')
                # Check every word in entity1, and add a new row triplet if it is present in entity1
                for entity in e2_sentence:
                    if entity in entity_set:
                        _ = (entities[entity], j[0], j[1], entities[j[2]], j[2])
                        triplet.add(_)
                        added = True
                if not added:
                    _ = ('O', j[0] ,j[1] ,'O', entities[j[2]])
                    triplet.add(_)
        
        # Convert the pandas dataframe into csv
        processed_pd = pd.DataFrame(list(triplet),columns=['Type', 'Entity1', 'Relationship', 'Type', 'Entity2'])
        processed_pd.to_csv(path + 'data/result/' + file.split("/")[-1].split(".")[0] + '.csv', encoding='utf-8', index=False)

        print('\nInput -->', path +"data/output/kg/"+file_name+".txt-out.csv")
        print('Output -->', path + 'data/result/' + file.split("/")[-1].split(".")[0] + '.csv')
        print("Processed " + file.split("/")[-1])

    print("\nFiles processed and saved")

    print('\n--------------------------------------------------------------------------------\n')
    