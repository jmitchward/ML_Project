import program_manager

object_list = []


print("Welcome.")
print("Creating the first machine learning object...")
object_list.append(program_manager.pg_manage())
for ml_object in object_list:
    ml_object.menu()
