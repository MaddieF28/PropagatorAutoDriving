-- Sample schema and data for SQL propagator demos
-- Load with: Catalog.from_sql_file('sample_data.sql')

CREATE TABLE customers (
    id INT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    city TEXT
);

INSERT INTO customers VALUES
    (1, 'Alice', 'NYC'),
    (2, 'Bob', 'LA'),
    (3, 'Carol', 'NYC'),
    (4, 'Dave', 'SF'),
    (5, 'Eve', 'LA');

CREATE TABLE orders (
    order_id INT PRIMARY KEY NOT NULL,
    cust_id INT NOT NULL,
    product_id INT NOT NULL,
    amount FLOAT NOT NULL
);

INSERT INTO orders VALUES
    (101, 1, 1, 150.0),
    (102, 1, 2, 50.0),
    (103, 2, 1, 200.0),
    (104, 3, 3, 75.0),
    (105, 3, 1, 300.0),
    (106, 4, 2, 120.0),
    (107, 5, 3, 90.0),
    (108, 1, 3, 180.0);

CREATE TABLE products (
    product_id INT PRIMARY KEY NOT NULL,
    pname TEXT NOT NULL,
    price FLOAT NOT NULL
);

INSERT INTO products VALUES
    (1, 'Widget', 25.0),
    (2, 'Gadget', 15.0),
    (3, 'Doohickey', 30.0);
