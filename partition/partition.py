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
except Exception as e:
	print(str(e))
rows = session.execute("select * from users")
# round robin partition
total_disk = 4
partitions = {}
for i in range(4):
	partitions[i] = []
	
for id,row in enumerate(rows):
	partitions[id%total_disk].append((row.username,row.age))
print(partitions)
