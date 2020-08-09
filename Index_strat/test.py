from Functions.conn import db_upload

sql = "INSERT INTO daily_ai_indicators (metric, value) VALUES (%s, %s)"
val = ("delta_strategy", 123)
db_upload(sql,val)


# sql = "INSERT INTO test_data (id, name, fav_num) VALUES (%s, %s, %s)"
# val = (111, "briman", 22)
# db_upload(sql,val)

