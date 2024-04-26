import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QFileDialog, QVBoxLayout, QLabel, QWidget
import numpy as np
import pandas as pd
import joblib

#finished with working thing but only done with second video, start third: https://www.youtube.com/watch?v=FVpho_UiDAY&list=PLzMcBGfZo4-lB8MZfHPLTEHO9zJDDLpYj&index=3&ab_channel=TechWithTim

class FileExplorerApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        layout = QVBoxLayout()

        self.button = QPushButton('Open File Explorer', self)
        self.button.setFixedSize(200,40)
        self.button.clicked.connect(self.openFileExplorer)
        self.label = QLabel("No file chosen", self)

        layout.addWidget(self.button)
        layout.addWidget(self.label)

        self.setLayout(layout)
        self.setWindowTitle('File Explorer')
        self.show()


    def openFileExplorer(self):
        self.label.setText("Loading...")
        self.show()
    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;Python Files (*.py)", options=options)

        if fileName:
            print("\nSelected file:", fileName,"\n")
            converted_file = self.convert_clean(fileName) # generate labels
            self.label.setText("File should be opening :)")
            self.show()
            #self.close() # close file explorer button window

            # # open label gen file depending on os
            if sys.platform.startswith('win'):
                from os import startfile
                startfile(converted_file)
            elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
                import subprocess
                subprocess.Popen(['xdg-open', converted_file])


    # label generation method
    def convert_clean(self, new_path_type2):

        #Read excel file
        new_df_type2 = pd.read_excel(new_path_type2)

        # Full column headers
        full_column_headers = ['Sample', 'CPU Date', 'CPU Time', 'Site Time', 'Period Time', 'Protocol Type', 'Storage ID', 'First Beat ID', 'Last Beat ID', 'Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'RH', 'Tbox', 'Tbody', 'Sr', 'Phase']

        # Get ML column headers
        column_headers = ['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']

        #Extract features
        new_features_type2 = new_df_type2[['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']]

        #Convert to numpy file (maybe)
        np.save('new_data_train_type2.npy', new_features_type2)

        # Load the saved numpy file
        new_data_train_type2 = np.load('new_data_train_type2.npy', allow_pickle=True)

        # Make predictions on saved numpy file
        # Load the model from the file
        loaded_model_type2 = joblib.load('best_pleth_ml_model_type2.pkl')
        pred_type2 = loaded_model_type2.predict(new_data_train_type2)

        # Create new column to store the generated labels and add column name to column_headers list
        new_column = np.ones((new_data_train_type2.shape[0], 1))
        full_column_headers += ["Generated Labels"]

        # Append the new column
        modified_data = np.concatenate((new_data_train_type2, new_column), axis=1)

        # Fill the new column with generated labels
        modified_data[:,-1] = pred_type2

        # Create a new column to store the generated labels
        new_df_type2["Generated Labels"] = pred_type2

        # Save predicted labels to 'generatelabeltest.npy' file
        np.save('generatelabeltest.npy', pred_type2)

        # Load the generated labels
        generated_labels = np.load('generatelabeltest.npy')

        # Define label descriptions
        label_descriptions = {
            0: "Normal / Quiet Breath",
            1: "Sigh breath",
            2: "Sniffing breath",
            3: "Random Apnea",
            4: "Type II Apnea"
        }

        # Count the occurrences of each label
        label_counts = np.unique(generated_labels, return_counts=True)

        # Initialize total breath count
        total_breath_count = 0

        # Print the counts with descriptions
        for label, count in zip(label_counts[0], label_counts[1]):
            description = label_descriptions[int(label)]
            print(f"{description} count ({int(label)}): {count}")
            # Increment total breath count
            total_breath_count += count

        # Print total breath count
        print(f"Total breath count: {total_breath_count}")

        # Add a new column for generated labels to the original DataFrame
        new_df_type2["Generated Labels"] = generated_labels

        # Convert and export DataFrame to Excel with all columns
        excel_file_path = new_path_type2[:-5] + "_LABELGEN.xlsx" # change new file name
        print("\nSuccess! New labeled file:", excel_file_path, "\n")
        new_df_type2.to_excel(excel_file_path, index=False)
        return excel_file_path


if __name__ == '__main__':
    app = QApplication(sys.argv)
    angie = FileExplorerApp()
    sys.exit(app.exec_())