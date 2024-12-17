-- Insert sample data into BOOKS table
INSERT INTO BOOKS (BookID, Title, Author, ISBN, Category, Publisher, PublicationDate, Price) VALUES
(1, 'The Great Gatsby', 'F. Scott Fitzgerald', '9780743273565', 'Fiction', 'Scribner', '1925-04-10', 10.99),
(2, '1984', 'George Orwell', '9780451524935', 'Dystopian', 'Plume', '1949-06-08', 9.99),
(3, 'Clean Code', 'Robert C. Martin', '9780132350884', 'Programming', 'Prentice Hall', '2008-08-01', 49.99),
(4, 'The Lean Startup', 'Eric Ries', '9780307887894', 'Business', 'Crown Business', '2011-09-13', 26.00),
(5, 'Obsolete Book', 'Unknown Author', '0000000000000', 'Outdated', 'Old Publisher', '1900-01-01', 5.00);

-- Insert sample data into INVENTORY table
INSERT INTO INVENTORY (BookID, QuantityOnHand, ReorderLevel, Status, WarehouseLocation) VALUES
(1, 120, 50, 'Active', 'Warehouse A - Shelf 1'),
(2, 30, 40, 'Active', 'Warehouse A - Shelf 2'),
(3, 200, 100, 'Active', 'Warehouse B - Shelf 3'),
(4, 15, 20, 'Active', 'Warehouse B - Shelf 4'),
(5, 5, 0, 'Obsolete', 'Warehouse C - Shelf 5');

-- Insert sample data into INVENTORY_ADJUSTMENTS table
INSERT INTO INVENTORY_ADJUSTMENTS (AdjustmentID, BookID, AdjustmentDate, AdjustmentType, QuantityAdjusted, Reason) VALUES
(1001, 1, '2023-01-15', 'Purchase', 150, 'Initial stock purchase'),
(1002, 1, '2023-02-20', 'Sale', -30, 'Monthly sales'),
(1003, 1, '2023-03-22', 'Sale', -50, 'Promotional event'),
(1012, 1, '2023-07-20', 'Purchase', 50, 'Restock due to high demand'),
(1004, 2, '2023-02-10', 'Purchase', 50, 'Restock'),
(1005, 2, '2023-03-15', 'Sale', -20, 'Online sales'),
(1013, 2, '2023-08-25', 'Sale', -10, 'Summer sale'),
(1006, 3, '2023-01-05', 'Purchase', 300, 'Bulk purchase'),
(1007, 3, '2023-04-18', 'Sale', -100, 'Corporate sales'),
(1014, 3, '2023-09-30', 'Sale', -50, 'End of quarter sale'),
(1008, 4, '2023-02-25', 'Purchase', 25, 'Restock'),
(1009, 4, '2023-05-30', 'Sale', -10, 'Retail sales'),
(1015, 4, '2023-10-10', 'Purchase', 10, 'Minor restock'),
(1010, 5, '2023-01-01', 'Obsolescence', -10, 'Write-off obsolete stock'),
(1011, 5, '2023-06-15', 'Sale', -5, 'Clearance sale');
