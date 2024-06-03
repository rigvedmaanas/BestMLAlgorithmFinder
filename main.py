import tkinter.filedialog
from customtkinter import *
from CTkTable import CTkTable
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tkinter.messagebox import showerror
class ShowAccuracy(CTkFrame):
    def __init__(self, *args, best, accuracy, **kwargs):
        super().__init__(*args, **kwargs)

        self.lbl = CTkLabel(self, text=f"Most accurate ML algorithm for your dataset is: {best}", font=CTkFont(size=35))
        self.lbl.pack(padx=10, pady=10)
        temp = CTkFrame(self, fg_color="transparent")
        temp.pack(fill="both", expand=True)

        self.frame_create(temp, "Logistic Regression", accuracy["Logistic Regression"])
        self.frame_create(temp, "Categorical NB", accuracy["Categorical NB"])

        temp2 = CTkFrame(self, fg_color="transparent")
        temp2.pack(fill="both", expand=True)

        self.frame_create(temp2, "Decision Tree Classifier", accuracy["Decision Tree Classifier"])
        self.frame_create(temp2, "Support Vector Classifier", accuracy["Support Vector Classifier"])




    def frame_create(self, root, model, accu):
        t = CTkFrame(root)
        t.pack(fill="both", padx=10, pady=10, expand=True, side="left")
        fr = CTkFrame(t, width=300, height=300)
        fr.pack(padx=10, pady=10, expand=True)
        #t.configure(fg_color=fr.cget("fg_color"))

        lbl = CTkLabel(fr, text=model, font=CTkFont(size=25))
        lbl.pack(pady=10, padx=10)

        lbl2 = CTkLabel(fr, text=f"Accuracy: {accu}", font=CTkFont(size=15))
        lbl2.pack(pady=10, padx=10)






class TrainFrame(CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LABEL4 = CTkLabel(master=self, text="Input Values", anchor="w", font=CTkFont(size=20))
        self.LABEL4.pack(pady=(10, 10), fill="x", padx=10)

        self.FRAME5 = CTkFrame(master=self)
        self.FRAME5.pack(pady=(10, 10), expand=1, fill="both", padx=10)
        self.FRAME5.columnconfigure(tuple(range(5)), weight=1)
        self.FRAME5.rowconfigure(tuple(range(20)), weight=1)
        self.check_boxes = {}
        self.check_frame = None

    def on_click(self, head):
        if self.check_frame.optionmenu.get() == head:
            self.check_boxes[head].deselect()
    def create_radiobutton(self, row, column, text):
        CHECKBOX6 = CTkCheckBox(master=self.FRAME5, text=text, command=lambda text=text: self.on_click(text))
        CHECKBOX6.grid(row=row, column=column, pady=10, padx=10)
        self.check_boxes[text] = CHECKBOX6
        CHECKBOX6.toggle()


class PreProcessor(CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LABEL4 = CTkLabel(master=self, text="Preprocessing", anchor="w", font=CTkFont(size=20))
        self.LABEL4.pack(pady=(10, 10), fill="x", padx=10)

        self.FRAME5 = CTkFrame(master=self)
        self.FRAME5.pack(pady=(10, 10), expand=1, fill="both", padx=10)
        self.FRAME5.columnconfigure(tuple(range(5)), weight=1)
        self.FRAME5.rowconfigure(tuple(range(20)), weight=1)
        self.preprocess_boxes = {}

    def add_preprocessor(self, row, column, text):
        PreProcessorFrame = CTkFrame(master=self.FRAME5)
        PreProcessorFrame.grid(row=row, column=column, padx=10, pady=10)
        v = StringVar(self, "2")
        LABEL22 = CTkLabel(master=PreProcessorFrame, text=text, anchor="w", font=CTkFont(size=15))
        LABEL22.pack(pady=(10, 5), fill="x", padx=10)

        RADIOBUTTON21 = CTkRadioButton(master=PreProcessorFrame, text="None", variable=v, value=0)
        RADIOBUTTON21.pack(padx=(10, 10), fill="x", pady=10)

        RADIOBUTTON23 = CTkRadioButton(master=PreProcessorFrame, text="LabelEncoder", variable=v, value=1)
        RADIOBUTTON23.pack(padx=(10, 10), fill="x")

        RADIOBUTTON25_copy = CTkRadioButton(master=PreProcessorFrame, text="StandardScaler", variable=v, value=2)
        RADIOBUTTON25_copy.pack(pady=(10, 10), fill="x", padx=10)
        RADIOBUTTON25_copy.select()

        self.preprocess_boxes[text] = v

class TestFrame(CTkFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LABEL4 = CTkLabel(master=self, text="Output Value", anchor="w", font=CTkFont(size=20))
        self.LABEL4.pack(pady=(10, 10), fill="x", padx=10)

        self.FRAME5 = CTkFrame(master=self)
        self.FRAME5.pack(pady=(10, 10), expand=1, fill="both", padx=10)
        self.check_frame = None

    def on_click(self, head):
        if self.check_frame.check_boxes[head].get() == 1:
            self.check_frame.check_boxes[head].deselect()

    def create_optionmenu(self, values):
        self.optionmenu = CTkOptionMenu(master=self.FRAME5, values=values, command=lambda text: self.on_click(text))
        self.optionmenu.pack(fill="x", expand=True, padx=10, pady=10)
        self.optionmenu.set(values[-1])


class App(CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("1900x1000")
        self.SCROLLABLEFRAME0 = CTkScrollableFrame(master=self, orientation="vertical")
        self.SCROLLABLEFRAME0.pack(expand=True, fill="both")

        self.LABEL1 = CTkLabel(master=self.SCROLLABLEFRAME0, text="Best Model Finder", font=CTkFont(size=30))
        self.LABEL1.pack(pady=(10, 0))

        self.table_frame = CTkFrame(master=self.SCROLLABLEFRAME0, height=366)
        self.table_frame.pack_propagate(False)
        self.table_frame.pack(pady=(20, 20), fill="x", padx=20)

        self.open_dataset_btn = CTkButton(self.table_frame, text="Open Dataset", corner_radius=3, height=30, command=self.open_dataset)
        self.open_dataset_btn.pack(expand=True)

        self.train_value_frame = TrainFrame(master=self.SCROLLABLEFRAME0, height=200, fg_color=("gray86", "#2e2e2e"))
        self.train_value_frame.pack(pady=(20, 20), fill="x", padx=20)

        self.test_value_frame = TestFrame(master=self.SCROLLABLEFRAME0, height=200, fg_color=("gray86", "#2e2e2e"))
        self.test_value_frame.pack(pady=(20, 20), fill="x", padx=20)

        self.test_value_frame.check_frame = self.train_value_frame
        self.train_value_frame.check_frame = self.test_value_frame

        self.FRAME16_copy = CTkFrame(master=self.SCROLLABLEFRAME0, fg_color=("gray86", "#2d2e2d"))
        self.FRAME16_copy.pack(pady=(20, 20), fill="x", padx=20)

        self.LABEL17_copy = CTkLabel(master=self.FRAME16_copy, text="Test Size - 0.2", anchor="w", font=CTkFont(size=20))
        self.LABEL17_copy.pack(pady=(10, 10), fill="x", padx=10)

        self.test_size_val = CTkSlider(master=self.FRAME16_copy, from_=0, to=1, number_of_steps=100, orientation="horizontal", command=lambda e: self.LABEL17_copy.configure(text=f"Test Size - {round(e, 2)}"))
        self.test_size_val.pack(pady=(10, 10), fill="x", padx=(10, 0))
        self.test_size_val.set(0.2)

        self.preprocess = PreProcessor(master=self.SCROLLABLEFRAME0, fg_color=("gray86", "#2d2e2d"))
        self.preprocess.pack(pady=(20, 20), fill="x", padx=20)


        self.BUTTON25 = CTkButton(master=self.SCROLLABLEFRAME0, text="Train", height=30, corner_radius=2, font=CTkFont(size=15), command=self.get_vals_to_train)
        self.BUTTON25.pack(pady=(20, 20), fill="x", padx=20)

    def open_dataset(self):
        file = filedialog.askopenfilename(filetypes=[("csv", ".csv")])
        if file != "":
            self.open_dataset_btn.destroy()
            self.dataset = pd.read_csv(file)

            self.dataset_array = [self.dataset.columns.values.tolist()] + self.dataset.head(10).values.tolist()
            self.headings = self.dataset.columns.values.tolist()
            self.table = CTkTable(self.table_frame, values=self.dataset_array)
            self.table.pack(expand=True, fill="both", padx=5, pady=5)

            self.test_value_frame.create_optionmenu(values=self.headings)

            column = 0
            row = 0
            for head in self.headings:

                self.train_value_frame.create_radiobutton(row=row, column=column, text=head)
                self.preprocess.add_preprocessor(row=row, column=column, text=head)
                column += 1
                if column > 5:
                    row += 1
                    column = 0

    def get_vals_to_train(self):
        try:
            input_values = self.train_value_frame.check_boxes
            new_input_values = {}
            for val in list(input_values.keys()):
                if input_values[val].get() == 1:
                    new_input_values[val] = input_values[val]
            self.x = pd.concat([self.dataset[key] for key in list(new_input_values.keys())], axis=1)

            self.y = self.dataset[self.test_value_frame.optionmenu.get()]
            self.x_copy = self.x.copy()
            self.y_copy = self.y.copy()


            self.test_size = self.test_size_val.get()
            for x in list(self.preprocess.preprocess_boxes.keys()):
                if self.preprocess.preprocess_boxes[x].get() == "1":
                    if x in self.x:
                        self.x[x] = self.do_label_encoder(self.x[x])
                    elif x in self.y:
                        self.y[x] = self.do_label_encoder(self.y[x])
                    if x in self.x_copy:
                        self.x_copy[x] = self.do_label_encoder(self.x_copy[x])
                    elif x in self.y_copy:
                        self.y_copy[x] = self.do_label_encoder(self.y_copy[x])
                elif self.preprocess.preprocess_boxes[x].get() == "2":
                    if x in self.x:
                        self.x[x] = self.do_standard_scaler(self.x[x].values.reshape(-1, 1))

                    elif x in self.y:
                        self.y[x] = self.do_standard_scaler(self.y[x].values.reshape(-1, 1))



            self.x = self.x.values
            self.y = self.y.values
            self.x_copy = self.x_copy.values
            self.y_copy = self.y_copy.values

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size)
            self.x_train_copy, self.x_test_copy, self.y_train_copy, self.y_test_copy = train_test_split(self.x_copy, self.y_copy, test_size=self.test_size)
            print(self.x_train.shape, self.y_train.shape)
            self.lr = self.logistic_regression(self.x_train, self.y_train)
            self.bt = self.bayes_theorem(self.x_train_copy, self.y_train_copy)
            self.sv = self.svm(self.x_train, self.y_train)
            self.dt = self.decision_tree_classifier(self.x_train, self.y_train)

            pred_lr = self.predict_and_get_accuracy_model(self.lr, self.x_test, self.y_test)
            pred_bt = self.predict_and_get_accuracy_model(self.bt, self.x_test_copy, self.y_test_copy)
            pred_sv = self.predict_and_get_accuracy_model(self.sv, self.x_test, self.y_test)
            pred_dt = self.predict_and_get_accuracy_model(self.dt, self.x_test, self.y_test)

            self.SCROLLABLEFRAME0.destroy()
            self.SCROLLABLEFRAME0._parent_frame.destroy()
            d = {"Logistic Regression": pred_lr, "Categorical NB": pred_bt, "Decision Tree Classifier": pred_dt, "Support Vector Classifier": pred_sv}
            d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
            print(d)
            self.pred = ShowAccuracy(self, best=list(d.keys())[-1], accuracy=d)
            self.pred.pack(fill="both", expand=True, padx=10, pady=10)
        except Exception as e:
            showerror("Error", e)
    def predict_and_get_accuracy_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        return accuracy_score(y_test, y_pred)




    def do_standard_scaler(self, val):
        sc = StandardScaler()
        return sc.fit_transform(val)

    def do_label_encoder(self, val):
        le = LabelEncoder()
        return le.fit_transform(val)

    def logistic_regression(self, x_train, y_train):
        lr = LogisticRegression()
        lr = lr.fit(x_train, y_train)
        return lr

    def bayes_theorem(self, x_train, y_train):
        nb_model = CategoricalNB()
        nb_model = nb_model.fit(x_train, y_train)
        return nb_model

    def svm(self, x_train, y_train):
        model = SVC(kernel="linear")
        model = model.fit(x_train, y_train)
        return model

    def decision_tree_classifier(self, x_train, y_train):
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        tree = tree.fit(x_train, y_train)
        return tree


set_default_color_theme("blue")
root = App()
root.title("ML Algorithm Finder")
root.configure(fg_color=['gray92', 'gray14'])
root.mainloop()
