package com.cslab.bigdata;

import java.util.Iterator;
import java.util.List;

import javax.jdo.PersistenceManager;
import javax.jdo.PersistenceManagerFactory;
import javax.jdo.Query;
import javax.jdo.Transaction;

import org.datanucleus.api.jdo.JDOPersistenceManagerFactory;
import org.datanucleus.metadata.PersistenceUnitMetaData;

public class JdoExp {
	PersistenceUnitMetaData pumd;
	PersistenceManagerFactory pmf;
	PersistenceManager pm;

	JdoExp() {
		this.pumd = new PersistenceUnitMetaData("testing", "RESOURCE_LOCAL", null);
		this.pumd.addClassName("com.cslab.bigdata.Product");
		this.pumd.addProperty("javax.jdo.option.ConnectionDriverName", "org.h2.Driver");
		this.pumd.addProperty("javax.jdo.option.ConnectionURL", "jdbc:h2:mem:mypersistence");
		this.pumd.addProperty("datanucleus.schema.autoCreateTables", "true");
		this.pumd.addProperty("datanucleus.query.jdoql.allowAll", "true");
		this.pmf = new JDOPersistenceManagerFactory(this.pumd, null);
		this.pm = this.pmf.getPersistenceManager();
	}

	void addData(String name, double price) {
		Transaction tx = pm.currentTransaction();
		tx.begin();
		Product product = new Product(name, price);
		pm.makePersistent(product);
		tx.commit();
	}

	void fetchData(String query) {
		Query q = pm.newQuery(query);
		List<Product> products = (List<Product>) q.execute();
		Iterator<Product> iter = products.iterator();
		System.out.print("\n");
		while (iter.hasNext()) {
			Product p = iter.next();
			System.out.printf("%d %s %f\n", p.id, p.name, p.price);
		}
		System.out.print("\n");
	}

	Long deleteQuery(String query) {
		Query q = pm.newQuery(query);
		return (Long) q.execute();
	}
	 
	protected void finalize() {
		this.pm.close();
		this.pmf.close();
	}

	public static void main(String[] args) {
		JdoExp program = new JdoExp();
		
		System.out.println("Adding items");
		program.addData("Tablet", 80.0);
		program.addData("Iphone2", 89990.0);
		program.addData("Iphone3", 89990.0);
		program.addData("Iphone", 100.0);

		System.out.println("All Items");
		program.fetchData("SELECT FROM " + Product.class.getName());

		System.out.println("Items with price > 80");
		program.fetchData("SELECT FROM " + Product.class.getName() + " WHERE price>80");

		System.out.println("Deleting items with price greater than 100");
		Long num_result = program.deleteQuery("DELETE FROM " + Product.class.getName() + " WHERE price > 100");
		System.out.printf("Deleted %d items",num_result);
		
		program.fetchData("SELECT FROM " + Product.class.getName());

		System.out.println("Updating tablet's name to Android");
		Product toupdate = program.pm.getObjectById(Product.class,1);
		toupdate.name = "Android";
		program.pm.makePersistent(toupdate);
		program.fetchData("SELECT FROM " + Product.class.getName());


	}
}
