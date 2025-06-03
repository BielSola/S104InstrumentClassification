import dataset_creation
import feature_extraction
import sample_data
import train_model



def main():
    """
    Main function to run the script
    """
    dt_path = 'G:/.shortcut-targets-by-id/17yphSXB2IgKWLJF-VDo9xJDWWM2e6mkH/S104/dataset'
    saraga = dataset_creation.create_dataset(dt_path)

    list_of_track_id = dataset_creation.get_number_of_tracks(2)
    train_model.train_model(list_of_track_id, what_to_predict='contains_vocal')

if __name__ == "__main__":
    main()