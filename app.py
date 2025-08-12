import pandas as pd
import customtkinter as ctk
from PCA_model import PCAModel
import threading
import matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class SimpleApp():
    def __init__(self):
        # Create main window
        self.root = ctk.CTk()
        
        self.setup_ui()

    def setup_ui(self):
        self.input_filepath=ctk.StringVar()
        self.output_folderpath=ctk.StringVar()
        self.inverse_filepath = ctk.StringVar()
        self.inverse_output = ctk.StringVar()

        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight= 1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        col_1 = ctk.CTkScrollableFrame(self.root)
        col_1.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        col_2 = ctk.CTkScrollableFrame(self.root)
        col_2.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        #calculators
        self.inverse_frame = ctk.CTkFrame(col_2)
        self.inverse_frame.grid(row=0, column=0)

        #inverse calculator
        self.inverse_label = ctk.CTkLabel(self.inverse_frame, text="please select input file")
        self.inverse_label.grid(row=0, column=0, columnspan = 2)
        self.inverse_input = ctk.CTkEntry(self.inverse_frame, textvariable=self.inverse_filepath)
        self.inverse_input.grid(row=1, column =0, padx=5)
        self.inversebrowse_button = ctk.CTkButton(self.inverse_frame, text="select file", command=lambda: self.browse_file(self.inverse_filepath, self.inverse_label))
        self.inversebrowse_button.grid(row=1, column =1, padx=5)
        self.inverse_button = ctk.CTkButton(self.inverse_frame, text="inverse transform", command=self.start_inverse_transform)
        self.inverse_button.grid(row=4, column=0, columnspan=2, padx=5)

        self.inverse_output_label = ctk.CTkLabel(self.inverse_frame, text="Select the output folder")
        self.inverse_output_label.grid(row=2, column=0, columnspan=2, pady=5)
        self.inverse_output_entry = ctk.CTkEntry(self.inverse_frame, textvariable=self.inverse_output)
        self.inverse_output_entry.grid(row=3, column=0, padx=5,pady=5)
        self.inversebrowse_button = ctk.CTkButton(self.inverse_frame, text="select file", command=lambda: self.browse_output(self.inverse_output, self.inverse_output_label))
        self.inversebrowse_button.grid(row=3,column=1, padx=5,pady=5)

       

        title=ctk.CTkLabel(col_1, text="PCA model retraining",font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, pady=20)
        
        self.input_label = ctk.CTkLabel(col_1, text="select your input file:")
        self.input_label.grid(row=1, column=0, pady=5)

        #row2 input
        self.input_frame = ctk.CTkFrame(col_1)
        self.input_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)
        self.file_entry = ctk.CTkEntry(self.input_frame, textvariable=self.input_filepath)
        self.file_entry.grid(row=0, column=0, padx=5)
        self.inputbrowse_button = ctk.CTkButton(self.input_frame, text="select file", command=lambda: self.browse_file(self.input_filepath, self.input_label))
        self.inputbrowse_button.grid(row=0, column=1, padx=5)
        
        #row3 outputlabel
        self.output_label = ctk.CTkLabel(col_1, text="Select the output folder")
        self.output_label.grid(row=3, column=0, pady=5)
        #row4 output
        self.output_frame = ctk.CTkFrame(col_1)
        self.output_frame.grid(row=4, column=0, sticky="nsew", padx=20, pady=20)
        
        self.output_entry = ctk.CTkEntry(self.output_frame, textvariable=self.output_folderpath)
        self.output_entry.grid(row=0, column=0, padx=5)
        self.outputbrowse_button = ctk.CTkButton(self.output_frame, text="select file", command=lambda: self.browse_output(self.output_folderpath, self.output_label))
        self.outputbrowse_button.grid(row=0, column=1, padx=5)

        #row5 trainbutton
        self.train_frame = ctk.CTkFrame(col_1)
        self.train_frame.grid(row=5, column=0, sticky="nsew", padx=20, pady=20)
        self.train_button = ctk.CTkButton(self.train_frame, text="train model", command=self.start_train_model)
        self.train_button.grid(row=0, column=0,rowspan =2,padx=5)
        self.components_entry = ctk.CTkEntry(self.train_frame, width=60, height=40)
        self.components_entry.grid(row=0, column=1,padx=5)
        self.comp_label = ctk.CTkLabel(self.train_frame, text="number of components")
        self.comp_label.grid(row=1, column=1,padx=5)

        #row6-7 specify failure
        self.specify_frame = ctk.CTkFrame(col_1)
        self.specify_frame.grid(row=6, column=0, sticky="nsew", padx=20, pady=10)
        
        # Configure grid weights for the frame columns
        self.specify_frame.columnconfigure(0, weight=1)
        self.specify_frame.columnconfigure(1, weight=1)
        
        # Left side: Failure rows
        self.failure_rows = ctk.CTkLabel(self.specify_frame, text="rows of failures (start:end)")
        self.failure_rows.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.failure_entry = ctk.CTkEntry(self.specify_frame)
        self.failure_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Right side: Healthy rows
        self.healthy_rows = ctk.CTkLabel(self.specify_frame, text="rows of healthy (start:end)")
        self.healthy_rows.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.healthy_entry = ctk.CTkEntry(self.specify_frame)
        self.healthy_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        #row7 terminal
        self.terminal = ctk.CTkTextbox(col_1, width=200, height=400, wrap="word", activate_scrollbars=True)
        self.terminal.grid(row=8, column=0, pady=5, sticky="nsew")

    def browse_file(self, stringvar, label):
        try:
            filepath = ctk.filedialog.askopenfilename(
                title = "select your training data",
                filetypes = [("Excel files", "*.xlsx"), ("csv files", "*.csv")]
                )
            if filepath:
                stringvar.set(filepath)
                label.configure(text = "file selected!")
        except Exception as e:
            self.terminal.insert("end", str(e))
            self.terminal.see("end")
    def browse_output(self, stringvar, label):
        filepath = ctk.filedialog.askdirectory(title = "Select your output folder")
        if filepath:
            stringvar.set(filepath)
            label.configure(text = "folder selected!")
    def disable_buttons(self):
        self.train_button.configure(state="disabled")
        self.inputbrowse_button.configure(state="disabled")
        self.outputbrowse_button.configure(state="disabled")
    def enable_buttons(self):
        self.train_button.configure(state="normal")
        self.inputbrowse_button.configure(state="normal")
        self.outputbrowse_button.configure(state="normal")
    
    def start_train_model(self):
        if not self.input_filepath.get():
            self.terminal.insert("end", "Please select an input file\n")
            self.terminal.see("end")
            return
        if not self.output_folderpath.get():
            self.terminal.insert("end", "Please select an output folder\n")
            self.terminal.see("end")
            return
        if not self.components_entry.get():
            self.terminal.insert("end", "Please select the number of components\n")
            self.terminal.see("end")
            return
        if not self.healthy_entry.get():
            self.terminal.insert("end", "Please select the healthy rows\n")
            self.terminal.see("end")
            return
        if not self.failure_entry.get():
            self.terminal.insert("end", "Please select the failure rows\n")
            self.terminal.see("end")
            return
        
        self.disable_buttons()
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    def train_model(self):
        filetype = self.input_filepath.get().split('.')[-1]
        try:
            if filetype =="xlsx":
                df = pd.read_excel(self.input_filepath.get(), header=None, usecols=range(0,4))
            elif filetype =="csv":
                df = pd.read_csv(self.input_filepath.get(), header=None, usecols=range(0,4))
            varnames = ["mean", 'standard deviation', 'slope', 'mean engine diff']
            self.model = PCAModel()
            failrows = [int(i) for i in self.failure_entry.get().split(':')]
            healthrows = [int(i) for i in self.healthy_entry.get().split(':')]
            self.terminal.insert("end", f"{failrows}, {healthrows}\n")
            self.terminal.see("end")
            
            output_df = self.model.train_pca(df, int(self.components_entry.get()), varnames, failrows, healthrows)
            outputpath = self.output_folderpath.get() + "/output.xlsx"
            output_df.to_excel(outputpath, index=False)
            self.terminal.insert("end", "Model trained successfully!\n")
            self.terminal.see("end")
            self.enable_buttons()
        except Exception as e:
            self.terminal.insert("end", str(e))
            self.enable_buttons()
    
    def start_inverse_transform(self):
        if not self.inverse_filepath.get():
            self.terminal.insert("end", "Please select an input file\n")
            self.terminal.see("end")
            return
        if not self.inverse_output.get():
            self.terminal.insert("end", "Please select an output file\n")
            self.terminal.see("end")
            return
        if not hasattr(self, 'model') or self.model is None:
            self.terminal.insert("end", "Please train a model first\n")
            self.terminal.see("end")
            return
        self.disable_buttons()
        try:
            filetype = self.inverse_filepath.get().split('.')[-1]
            if filetype =="xlsx":
                df = pd.read_excel(self.inverse_filepath.get(), header=None)
            elif filetype =="csv":
                df = pd.read_csv(self.inverse_filepath.get(), header=None)
            self.inverse_transform(df)
        except Exception as e:
            self.terminal.insert("end", f"{str(e)}\n")
            self.enable_buttons()
    def inverse_transform(self,df):
        try:
            
            output_df = self.model.inverse_transform(df)
            outputpath = self.inverse_output.get() + "/inverseoutput.xlsx"
            output_df.to_excel(outputpath, index=False)
            self.terminal.insert("end", "Inverse transform completed successfully!\n")
            self.terminal.see("end")
            self.enable_buttons()
        except Exception as e:
            self.terminal.insert("end", str(e))
            self.enable_buttons()
    def run(self):
        self.root.mainloop()
if __name__ == "__main__":
    app = SimpleApp()
    app.run()  