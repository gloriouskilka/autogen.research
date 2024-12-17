-- Create the BOOKS table
CREATE TABLE BOOKS (
    BookID INT PRIMARY KEY,
    Title VARCHAR(255) NOT NULL,
    Author VARCHAR(255),
    ISBN VARCHAR(20),
    Category VARCHAR(100),
    Publisher VARCHAR(255),
    PublicationDate DATE,
    Price DECIMAL(10, 2)
);

-- Create the INVENTORY table
CREATE TABLE INVENTORY (
    BookID INT PRIMARY KEY,
    QuantityOnHand INT NOT NULL,
    ReorderLevel INT NOT NULL,
    Status VARCHAR(50),
    WarehouseLocation VARCHAR(100),
    FOREIGN KEY (BookID) REFERENCES BOOKS(BookID)
);

-- Create the INVENTORY_ADJUSTMENTS table
CREATE TABLE INVENTORY_ADJUSTMENTS (
    AdjustmentID INT PRIMARY KEY,
    BookID INT NOT NULL,
    AdjustmentDate DATE NOT NULL,
    AdjustmentType VARCHAR(50),
    QuantityAdjusted INT NOT NULL,
    Reason VARCHAR(255),
    FOREIGN KEY (BookID) REFERENCES BOOKS(BookID)
);
