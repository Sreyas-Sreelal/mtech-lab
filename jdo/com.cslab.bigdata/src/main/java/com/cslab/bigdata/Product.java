package com.cslab.bigdata;

import javax.jdo.annotations.IdGeneratorStrategy;
import javax.jdo.annotations.PersistenceCapable;
import javax.jdo.annotations.Persistent;
import javax.jdo.annotations.PrimaryKey;

@PersistenceCapable
public class Product {
	@PrimaryKey
	@Persistent(valueStrategy = IdGeneratorStrategy.INCREMENT)
	int id;
	String name;
	double price = 0.0;
	
	public Product(String name,double price) {
		this.name = name;
		this.price = price;
	}

}
