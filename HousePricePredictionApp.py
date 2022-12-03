from tkinter import *
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

options = [
    "Linear Regression",
    "Lasso Regression",
    "Ridge Regression",
    "Random Forest",
    "Decision Tree"
]

def predict(model_choice, first_floor_sqft, garage_area, above_ground_living_area, overall_quality, total_basement_square_feet, prediction_label):
    df = pd.read_csv("./ManualPreprocessedAmesHousing.csv")
    X = df[["1st Flr SF", "Garage Area", "Gr Liv Area", "Overall Qual", "Total Bsmt SF"]]
    y = df["SalePrice"]
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Lasso Regression":
        model = Lasso(alpha=0.25)
    elif model_choice == "Ridge Regression":
        model = Ridge(alpha=0.25)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(bootstrap=True, max_features="auto",max_depth=17,min_samples_leaf=3,min_samples_split=10,n_estimators=51)
    elif model_choice == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=11, min_samples_leaf=6)
    model.fit(X, y)
    if(first_floor_sqft == "" or garage_area == "" or above_ground_living_area == "" or overall_quality == "" or total_basement_square_feet == ""):
        prediction_label.config(text="Please fill in all fields")
        return
    X = pd.DataFrame([[first_floor_sqft, garage_area, above_ground_living_area, overall_quality, total_basement_square_feet]], columns=["1st Flr SF", "Garage Area", "Gr Liv Area", "Overall Qual", "Total Bsmt SF"])
    y_pred = model.predict(X)
    y_pred = str(int(abs(y_pred[0])) * 1000) + " USD"
    prediction_label.config(text=(str(y_pred)))
    return y_pred

def pop_up_instructions(root):
    top = Toplevel(root)
    top.geometry("300x300")
    top.title("Instructions")
    instructions = Label(top, text="Quick Instructions", font=("Arial", 20))
    instructions.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    detailed_instructiosn = Label(top, text="1. Fill in all fields with the appropriate values\n2. Click 'Predict' to get the predicted price\n3. Click 'Clear' to clear all fields")
    detailed_instructiosn.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    average_house = Label(top, text="Average House Stats", font=("Arial", 20))
    average_house.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

    average_stats = Label(top, text="Average 1st Floor Square Feet: 1150\nAverage Garage Area: 500\nAverage Above Ground Living Area: 1500\nAverage Overall Quality: 6\nAverage Total Basement Square Feet: 1000")
    average_stats.grid(row=3, column=0, padx=10, pady=5)



def main():
    root = Tk()
    root.title("House Price Prediction App")
    title = Label(root, text="House Price Prediction App", font=("Arial", 20))
    title.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

    instruction_btn = Button(root, text="Instructions", command=lambda: pop_up_instructions(root))
    instruction_btn.grid(row=0, column=1, padx=10, pady=10, ipadx=30)
    
    first_floor_sqft_label = Label(root, text="First Floor Square Feet")
    first_floor_sqft_label.grid(row=1, column=0, padx = 20, pady = (10, 0))
    first_floor_sqft_entry = Entry(root, width=10)
    first_floor_sqft_entry.grid(row=1, column=1, padx = 20, pady = (10, 0))
    
    garage_area_label = Label(root, text="Garage Area")
    garage_area_label.grid(row=2, column=0, padx = 20, pady = (10, 0))
    garage_area_entry = Entry(root, width=10)
    garage_area_entry.grid(row=2, column=1, padx = 20, pady = (10, 0))

    above_ground_living_area_label = Label(root, text="Above Ground Living Area")
    above_ground_living_area_label.grid(row=3, column=0, padx = 20, pady = (10, 0))
    above_ground_living_area_entry = Entry(root, width=10)
    above_ground_living_area_entry.grid(row=3, column=1, padx = 20, pady = (10, 0))

    overall_quality_label = Label(root, text="Overall Quality")
    overall_quality_label.grid(row=4, column=0, padx = 20, pady = (10, 0))
    overall_quality_entry = Entry(root, width=10)
    overall_quality_entry.grid(row=4, column=1, padx = 20, pady = (10, 0))

    total_basement_square_feet_label = Label(root, text="Total Basement Square Feet")
    total_basement_square_feet_label.grid(row=5, column=0, padx = 20, pady = (10, 0))
    total_basement_square_feet_entry = Entry(root, width=10)
    total_basement_square_feet_entry.grid(row=5, column=1, padx = 20, pady = (10, 0))

    model_choice = StringVar()
    model_choice.set(options[0])
    
    model_choice_label = Label(root, text="Model Choice")
    model_choice_label.grid(row=6, column=0, padx = 20, pady = (10, 0))
    model_choice_dropdown = OptionMenu(root, model_choice, *options)
    model_choice_dropdown.grid(row=6, column=1, padx = 20, pady = (10, 0))
    
    predict_btn = Button(root, text="Predict", command=lambda: predict(model_choice.get(), first_floor_sqft_entry.get(), garage_area_entry.get(), above_ground_living_area_entry.get(), overall_quality_entry.get(), total_basement_square_feet_entry.get(), prediction_label))
    predict_btn.grid(row=7, column=0, columnspan=1, padx=10, pady=10, ipadx=50)

    clear_btn = Button(root, text="Clear", command=lambda: first_floor_sqft_entry.delete(0, END) or garage_area_entry.delete(0, END) or above_ground_living_area_entry.delete(0, END) or overall_quality_entry.delete(0, END) or total_basement_square_feet_entry.delete(0, END) or prediction_label.config(text=""))
    clear_btn.grid(row=7, column=1, columnspan=1, padx=10, pady=10)

    prediction = Label(root, text="House SalePrice Prediction is: ")
    prediction.grid(row=8, column=0, padx = 20, pady = (10, 40))
    prediction_label = Label(root, text="")
    prediction_label.grid(row=8, column=1, padx = 20, pady = (10, 40))

   

    root.mainloop()


if __name__ == "__main__":
    main()
