-- Create the BOOKS table
CREATE TABLE BOOKS (
    BookID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title VARCHAR(255) NOT NULL,
    Author VARCHAR(255),
    ISBN VARCHAR(20) UNIQUE,
    Category VARCHAR(100),
    Publisher VARCHAR(255),
    PublicationDate DATE,
    Price DECIMAL(10, 2) NOT NULL,
    Cost DECIMAL(10, 2) NOT NULL DEFAULT 0.00
);

-- Create the SUPPLIERS table
CREATE TABLE SUPPLIERS (
    SupplierID INTEGER PRIMARY KEY AUTOINCREMENT,
    SupplierName VARCHAR(255) NOT NULL,
    ContactInfo VARCHAR(255),
    ReliabilityScore DECIMAL(3,1) DEFAULT 5.0, -- 1.0 to 5.0 scale
    DeliveryTimeDays INT -- Average delivery time in days
);

-- Create the INVENTORY table
CREATE TABLE INVENTORY (
    BookID INTEGER PRIMARY KEY,
    QuantityOnHand INT NOT NULL DEFAULT 0,
    ReorderLevel INT NOT NULL DEFAULT 0,
    Status VARCHAR(50) NOT NULL DEFAULT 'Active',
    WarehouseLocation VARCHAR(100),
    FOREIGN KEY (BookID) REFERENCES BOOKS(BookID) ON DELETE CASCADE
);

-- Create the INVENTORY_ADJUSTMENTS table
CREATE TABLE INVENTORY_ADJUSTMENTS (
    AdjustmentID INTEGER PRIMARY KEY AUTOINCREMENT,
    BookID INT NOT NULL,
    AdjustmentDate DATE NOT NULL,
    AdjustmentType TEXT NOT NULL CHECK(AdjustmentType IN ('Purchase', 'Sale', 'Obsolescence')),
    QuantityAdjusted INT NOT NULL,
    Reason VARCHAR(255),
    SupplierID INT, -- Allows NULL for non-purchase adjustments
    FOREIGN KEY (BookID) REFERENCES BOOKS(BookID) ON DELETE CASCADE,
    FOREIGN KEY (SupplierID) REFERENCES SUPPLIERS(SupplierID) ON DELETE SET NULL
);

-- Create the RETURNS table
CREATE TABLE RETURNS (
    ReturnID INTEGER PRIMARY KEY AUTOINCREMENT,
    BookID INT NOT NULL,
    ReturnDate DATE NOT NULL,
    QuantityReturned INT NOT NULL,
    Reason VARCHAR(255),
    FOREIGN KEY (BookID) REFERENCES BOOKS(BookID) ON DELETE CASCADE
);
