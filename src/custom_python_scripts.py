import os 

def rename_all_output_files(folder="../output_save"):
    # renames all the files in the output_save folder by 
    #removing the random generated prefix number 
    
    for filename in os.listdir(folder):
#         # rename() function will 
#         # rename all the files 
        os.rename(folder+'/'+filename, folder+'/'+filename[11:])