import core.dataset_creation as dataset_creation
import core.feature_extraction as feature_extraction
import core.sample_data as sample_data
import core.train_model as train_model
import core.run_pipeline as run_pipeline
import core.feature_selection as feature_selection



def main():
    """
    Main function to run the script
    """
    #dt_path = 'G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset'
    #saraga = dataset_creation.create_dataset(dt_path)

    #list_of_track_id = dataset_creation.get_number_of_tracks(20)
    train_model.train_model_csv("Data_Training.csv", what_to_predict='contains_mridangam')
    #feature_selection.select_features_from_csv(csv_file='Data_Training.csv', target='contains_mridangam')

if __name__ == "__main__":
    main()
    
