from cassandra.cluster import Cluster
cluster = Cluster()
session = cluster.connect('sreyas')

session.execute("Create table if not exists users(username text primary key,age int);")
try:
	session.execute("insert into users(username,age) values('sreyas',24);")
	session.execute("insert into users(username,age) values('robin',12);")
	session.execute("insert into users(username,age) values('yedhu',72);")
	session.execute("insert into users(username,age) values('joshua',32);")
	session.execute("insert into users(username,age) values('roshan',22);")
	session.execute("insert into users(username,age) values('jiby',24);")
	session.execute("insert into users(username,age) values('jimmy',42);")
	session.execute("insert into users(username,age) values('hilda',24);")
	session.execute("insert into users(username,age) values('sandra',22);")
	session.execute("insert into users(username,age) values('merin',22);")
	
	def execute_and_print(query):
		if query != "":
			session.execute(query)
		print("\nOutput")
		rows = session.execute("select * from users")
			
		for row in rows:
			print(row.username,row.age)
		print()
	
	execute_and_print("")
	print("setting yedhu's age to 25")
	execute_and_print("update users set age=25 where username='yedhu'")
	print("Deleting yedhu");
	execute_and_print("delete from users where username='yedhu'");
	print("Deleting entire table");
	session.execute("drop table users");
	
except Exception as e:
	print(str(e))

