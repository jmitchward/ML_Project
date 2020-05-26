# ML_Project

- [create_db](#create_db)
  - Create and organize a database
- [db_manage](#db_manage)
  - Change a database
- [pg_manage](#pg_manage)
  - Move around a database
- [predict_manage]()
  - Predict new conclusions from a database	
  
## Description
The idea of this program is to automate as much as possible in the process of producing results. This is said with the knowledge that the most important part of machine learning is understanding both the data you are putting in and the results you are producing. Without proper understanding of a dataset the results produced cannot be fully understood. That being said the impetus for this project was an idea for a senior thesis project that highlighted different algorithms used for supervised machine learning. I found a dataset at the UC Irvine Machine Learning Repository and wrote the initial code very rigidly for that dataset. The for loops were written exactly to the length of the dataset and the number of features. Once that project was concluded (with an A) I tossed and turned looking for a project to expand my knowledge base a little and become more familiar with python. 
   
Ideally all that needs to be provided is the dataset, which for now is added manually into the code base itself. The program will load the dataset and prompt the user to begin manipulating the dataset and then run that dataset through any of the three algorithms. The algorithms chosen were highlighted for their applications with supervised learning. The dataset used initially was created specifically for supervised learning with a binary classifier. The three written in this repo are: Naive Bayes, Logistic Regression, Decision Tree. Naive Bayes and Logistic Regression algorithms are both fully incorporated and ready for sure. The Decision Tree algorithm is still a work in progress and very slow, so more attention will be needed once I feel comfortable with where the rest of the program is at. The implementation of these algorithm align with the standard implementations found online so an explanation of the process behind these algorithms need not be explained here. 

The structure of the manager system is a essentially three separate files. The concept for these three files is to define three fundamental methods and isolate those methods within those files. There is an inheritance structure starting with create_database as the parent, database_manager as its child, then program_manager as the child of database_manager. This is used to ensure that all the variables and methods needed are available as each file is accessed interchangeably. The concept for the structure is to allow for multiple instances to be created, save, loaded, and used collectively. It is the idea to run an algorithm on a particular dataset then run the same or another algorithm on a different dataset. Both would return a binary result that can then be used in a new, derivative dataset that can then be used on its own to new conclusion. 

## Use Case

[ ] Under Construction

## Structure

### [create_db](create_database.py)

This class consists of all functions needed to retrieve and establish the desired database. It is the root/parent of database_manager.
		
**Variables**

- skip_check - String - Used as a check for several functions to ensure that no extra time is spent formatting data that has already been formatted
- train_class - List -  Used to store the classifier values for the train data, which is typically stored as a column/feature of the actual train data file.
- test_class - List - Used to store the classifier value for the test data, which is typically stored as a column/feature of the actual test data file.
- feature_names - List - Used to store user defined column/feature names
- features - List - Used to store three separate values. features[0] stores a list of categorical features, features[1] stores a list of numerical features, features[2] stores the feature/column value where the classifier can be found. 
- feature_values - Dictionary - Used to store the values of each feature in the dataset. This is done to ensure that no information is lost when the dataset is formatted for the algorithms. 
- train_data - Dataframe - The default declaration for the program when it is started. This is largely for testing purposes. 
- test_data-  Dataframe - The default declaration for the program it is started. This is largely for testing purposes. 
 - data - Dataframe - By default a copy of the train_data dataframe. This variable is used as the dataframe throughout the program. This was used to ensure that all the functions do not have to be written twice for each dataset. 
 
**Functions**
 
- [learning_method](create_database.py#L30)
  - Function called by database_manager.setup_dataset. 
  - Used to determine which method of learning will be taking place. As of now only supervised learning method is written as those are the only algorithms available. Once a selection is made it then goes forward to call supervised_learning. This function loops if yes is not answered to the initial prompt. 
  - **Variables**:
    - data_type - String -  User prompt to indicate what kind of learning will be taking place. This is temporarily a false choice. 
    - classifier - String -  User prompt to indicate where in the dataset the classifier is stored. This is subsequently type cast as an int and passed on. 
    - features - List - This variable is used in the function call for supervised_learning.
- [supervised_learning](create_database.py#L41)
    - Function called by learning_method. 
    - Used to determine which features/columns are categorical and which are numerical. This is done using a simple type check, iterating over every value of every feature. If the values are all strings, is is categorical. There is a simpler way to do this in terms of scale, but for now this is the implementation. Additionally the function removes the classifier from which ever list it has been added to. Function returns its variables to the function calling it. 
    - **Variables**
      - categorical - List - Used to store those feature/column values that are determined to be categorical. 
      - numerical - List - Used to store the remaining feature/column values that are not deemed categorical. This is done to reduce the computation time for this function. The majority of features will typically be numerical therefore it is easier to find those that are not then remove them from the initial list.
- [get_dataset](create_database.py#L81)
    - Function called by program_manager.menu_select. Calls program_manager.format_chain and program_manager.menu. 
    - Prompts the user to enter the URL of a new, undefined datasets into the program. If the desired dataset is already split into a training and testing set, they are imported separately and formatted. Currently this presents a false choice as the split automatically choice is not written. Once the desired outcome is reach, the function ends with a call to the menu function of the program_manager to move further along with the process. 
    - **Variables**
      - train_data - Dataframe - The initial definition for the program when this function is called. It is used as the first definition for the ubiquitous data variable used throughout the program, this is done to avoid passing parameters into functions that do not need to be. Data is rigidly defined between train_data and test_data at times appropriate to the progress of the program. Ideally, anyway. 
      - test_data - Dataframe - The initial definition for the program when this function is called. It is used as the first definition for the ubiquitous data variable used throughout the program, this is done to avoid passing parameters into functions that do not need to be. Data is rigidly defined between train_data and test_data at times appropriate to the progress of the program. Ideally, anyway. 
      - data - Dataframe - The initial definition for the program when this function is called. This variable is used as the dataframe throughout the program. This was used to ensure that all the functions do not have to be written twice for each dataset.
- [name_data](create_database.py#L107)
  - Function called by program_manager.run_ml_fn, which is the function within the menu operation for calling this function.
  - It is responsible for providing user defined column/feature names.
  - **Variables**
      - column_names - String -  Used to store user input providing the aforementioned column/feature name, which is then stored in the class defined feature_names.
- [load_instance](create_database.py#L127)
  - Function called program_manager.menu. 
  - Returns the object stored at the file_path parameter given to the function. As the program runs, it will save or prompt the user to save both the dataset structure as well as each of the algorithms ran. Objects loaded are pickles, which is the weirdest sentence I’ve ever written.
  - **Variables**
    - saved_dataset - object - The variable used to store and return the loaded file. 
- [save_insance](create_database.py#L132)
  - Function called by program_manager.manage_dataset, program_manager.run_ml_dt, program_manager.run_ml_fn, program_manager.run_ml_lr, program_manager.run_ml_nb. 
  - Stores the given object in the given file path, which is declared in constructorof the program manager.
  - **Variables**
    - None
    
[Top](#ML_Project)

### [db_manage](database_manager.py)

This class consists of all functions needed to access and change the desired database. It is the child of create_db and the parent of the program_manager. It therefore inherits all functions and initialized variables. 

**Variables**
  - None.
  
**Functions**

- [setup_dataset](database_manager.py#L11)
  - Function called by format_chain. Calls create_db.learning_method. 
  - This is done to control the number of access points to the parent class. The functions within the create_db class cascade from this function call. Any additional calls needed to the parent class will take place here. 
  - **Variables**
    - None
- [format_chain](database_manager.py#L14)
  - Function called by program_manager.manage_dataset, program_manager.run_ml_dt, program_manager.run_ml_lr, program_manager.run_ml_nb. Calls setup_dataset, standardize_data, format_data. 
  - Responsible for moving through all of the indepdent formatting functions within the class. 
  - **Variables**
    - None
- [encode_data](database_manager.py#L23)
  - Function called by format_data.
  - Responsible for encoding features/columns in integer form. This is done using categorical codes, which are essentially index values for each unique value of each feature/column. 
  - **Variables**
    - None
- [format_data](database_manager.py#L40)
  - Function called by format_chain. Calls encode_data, normalize_data
  - Responsible for encoding the dataset, stripping out the classifier and then removing the classifier.
  - **Variables**
    - classifiers - List - The classifiers for the dataset, pulled out of the dataset by the create_db class.
    - data - Dataframe - Removes the classifier from the data now that the classifiers are stored separately for processing.
- [normalize_data](database_manager.py#L63)
  - Function called by format_data. 
  - Responsbile for applying the normalization forumla to the data
  - **Variables**
    - list_of_maxs - List - The maximum values of each column/feature. This is a function of the pandas dataframe library.
    - list_of_mins - List - The minimum values of each column/feature. This is a fuction of the pnadas dataframe library.
- [standardize_data](database_manager.py#L71)
  - Function called by program_manager.run_ml_lr, program_manager.run_ml_nb
  - Responsible for applying the standardization formula to the data. 
  - **Variables**
    - None
- [prune_data](database_manager.py#L76)
  - Function called by program_manager.menu
  - Responsible for removing features/columns that need to be removed from the dataset for a multitude of reasons. Iterates over each feature/column in the dataset and prompts the user to remove that feature/column.
  - **Variables**
    - None
- [backup_database](database_manager.py#L92)
  - Function is not currently in use.

[Top](#ML_Project)

### [pg_manage](program_manager.py)

This class consists of all functions needed for running and moving throughout the program and its functions. It is the child of database_manager, which is the child of create_database class. It therefore inherits all functions and initialized variables.

**Variables**
  - ml_instance - String - Used to store the desired path where the dataset object can be accessed.
  - nb_path - String - Used to store the desired path where the Naive Bayes object can be accessed 
  - lr_path - String - Used to store the desired path where the Logistic Regression object can be accessed. 
  - dt_path - String - Used to store the desired path where the Decision Tree object can be accessed.
  
**Functions**

- [manage_dataset](program_manager.py#L25)
  - Called by menu_select and itself. Calls format_chain, menu
  - Responsible for initiating the formatting process through variable definitions and function calls to the database_manager. Prompts the user to select which dataset they would like to format. Function loops until a valid selection is made or a return request is made. Skip_check is changed to 'Yes' to indicate that the dataset has been formatted.
  - **Variables**
    - to_format - String - User input for the format request.
    - data - Dataframe - The default dataset is set to the present dataset and formatted
    - train_data - Dataframe - Redefined after formatting to ensure that the default dataset can be used for the test data.
    - train_class - List - Stores the previously defined and retrieved classifiers for the training set. 
    - test_data - Dataframe - Redefined after formatting to ensure that the default dataset can be used for the train data.
    - test_class - List - Stores the previously defined and retrieved classifiers for the test set.
- [run_ml_fn](program_manager.py#L72)
  - Called by menu_select. Calls name_data, save_instance
  - Responsible for running the name_data function
  - **Variables**
    None
- [run_ml_lr](program_manager.py#L77)
  - Called by menu_select. Calls format_chain, logistic_regression, save_instance
  - Responsible for ensuring the dataset is formatted and then running the logistic regression function. The resulting object is then stored in the path stored in lr_path.
  - **Variables**
    - lr_instance - Object - The returned object from the logistic regression function
- [run_ml_dt](program_manager.py#L85)
  - Called by menu_select. Calls format_chain, decision_tree, save_instance
  - Responsible for ensuring the dataset is formatted and then running the decision tree function. The resulting object is then stored in the path stored in dt_path.
  - **Variables**
    - dt_instance - Object - The returned object from the decision tree function
- [run_ml_nb](program_manager.py#L93)
  - Called by menu_select. Calls format_chain, naive_bayes, save_instance
  - Responsible for ensuring the dataset is formatted and then running the naive bayes function. The resulting object is then stored in the path stored in dt_path.
  - **Variables**
    - nb_instance - Object - The returned object from the naive bayes function
- [run_predictions](program_manager.py#L100)
  -Called by menu_select. Calls predict_manage
  - Responsible for running the prediction portion of the program. Returns a prediction in the form of 0 or 1.
  - **Variables**
    - algo - String - User selection for the algorithm to be used in the forthcoming prediction. 
- [menu](program_manager.py#L111)
  - Called by the constructor, manage_dataset, menu_select, run_ml_dt, run_ml_fn, run_ml_lr, run_ml_nb, run_predictions. Calls menu_select.
  - Responsible for presenting the user with a list of all possilbe operations that can be preformed. 
  - **Variables**
    - next_choice - String - User input that is passed to the menu_select function for a decision. 
- [menu_select](program_manager.py#L127)
  - Called by menu. Calls get_dataset, manage_dataset, prune_data, menu, run_ml_fn, run_ml_lr, run_ml_nb, run_ml_dt, run_predictions, load_state, menu. 
  - Responsible for executing user input and navigating through the program. Will recall the menu function if a choice is not made or if the user indciates 'exit'.
  - **Variables**
    - None
    
[Top](#ML_Project)
    


  
  
    
  





