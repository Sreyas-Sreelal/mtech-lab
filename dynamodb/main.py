import boto3
import os
import json

os.environ["AWS_DEFAULT_REGION"] = "us-west"
os.environ["AWS_Access_Key_ID"] = "fakeMyKeyId"
os.environ["AWS_Secret_Access_Key"] = "fakeSecretAccessKey"
client = boto3.resource('dynamodb', endpoint_url="http://localhost:8000")

try:
    table = client.create_table(
        TableName='CSLAB',
        KeySchema=[
            {
                'AttributeName': 'name',
                'KeyType': 'HASH',
            },
            {
                'AttributeName': 'id',
                'KeyType': 'RANGE',
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'id',
                'AttributeType': 'N',
            },
            {
                'AttributeName': 'name',
                'AttributeType': 'S',
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        }
    )
except boto3.exceptions.botocore.errorfactory.ClientError:
    print("Table already exists so skipping")
    table = client.Table("CSLAB")
print("TABLE STATUS:",table.table_status)

data = json.load(open('data.json'))
for item in data:
    print("Inserting",item)
    print(table.put_item(Item=item))

print("\nView all data")
for data in table.scan()['Items']:
    print(data)

print("\nSelect data with primary keys name and id with Yedhu and 2 respectively")
response = table.get_item(Key={"id":2,"name":"Yedhu"})['Item']
print("Result:",response)

print("\nSelect data with age<18")
response = table.scan(
    FilterExpression='age<:eighteen',
    ExpressionAttributeValues = {
        ":eighteen":18
    }
)["Items"]
print("Result:",response)

print("\nUpdate robin age to 24")
respone = table.update_item(
    Key={
        "id":3,
        "name":"Robin"
    },
    UpdateExpression="set age=:value",
    ExpressionAttributeValues={
        ":value":24
    }
)
print("Updated")
response = table.get_item(Key={"id":3,"name":"Robin"})['Item']
print("Result:",response)

print("\nDelete data with id and name, 4 and Jiby respectively")
response = table.delete_item(Key={"id":4,"name":"Jiby"})
print("Result:",response)
print("View all data")
for data in table.scan()['Items']:
    print(data)
print("\nDeleting table")
table.delete()
