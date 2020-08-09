import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="a3d"
)


# def what_does_the_fox_say():
#   print("gimme some money")


# sql = "INSERT INTO test_data (id, name, fav_num) VALUES (%s, %s, %s)"
# val = (10, "briman", 21)
def db_upload(sql, val):
  mycursor = mydb.cursor()
  mycursor.execute(sql, val)
  mydb.commit()
  print(mycursor.rowcount, "record inserted.")