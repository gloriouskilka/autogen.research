-- Insert sample data into BOOKS table
INSERT INTO BOOKS (BookID, Title, Author, ISBN, Category, Publisher, PublicationDate, Price, Cost) VALUES
(1, 'The Great Gatsby', 'F. Scott Fitzgerald', '9780743273565', 'Fiction', 'Scribner', '1925-04-10', 10.99, 6.50),
(2, '1984', 'George Orwell', '9780451524935', 'Dystopian', 'Plume', '1949-06-08', 9.99, 5.00),
(3, 'Clean Code', 'Robert C. Martin', '9780132350884', 'Programming', 'Prentice Hall', '2008-08-01', 49.99, 25.00),
(4, 'The Lean Startup', 'Eric Ries', '9780307887894', 'Business', 'Crown Business', '2011-09-13', 26.00, 13.00),
(5, 'Obsolete Book', 'Unknown Author', '0000000000000', 'Outdated', 'Old Publisher', '1900-01-01', 5.00, 2.00);

-- Insert sample data into SUPPLIERS table
INSERT INTO SUPPLIERS (SupplierName, ContactInfo, ReliabilityScore, DeliveryTimeDays) VALUES
('Supplier A', 'contactA@example.com', 4.5, 7),
('Supplier B', 'contactB@example.com', 4.8, 5),
('Supplier C', 'contactC@example.com', 3.9, 10);

-- Insert corresponding inventory data
INSERT INTO INVENTORY (BookID, QuantityOnHand, ReorderLevel, Status, WarehouseLocation) VALUES
(1, 120, 50, 'Active', 'Warehouse A - Shelf 1'),
(2, 30, 40, 'Active', 'Warehouse A - Shelf 2'),
(3, 200, 100, 'Active', 'Warehouse B - Shelf 3'),
(4, 15, 20, 'Active', 'Warehouse B - Shelf 4'),
(5, 5, 0, 'Obsolete', 'Warehouse C - Shelf 5');

-- Insert sample data into INVENTORY_ADJUSTMENTS table
-- Assuming SupplierIDs: Supplier A (1), Supplier B (2), Supplier C (3)
INSERT INTO INVENTORY_ADJUSTMENTS (AdjustmentID, BookID, AdjustmentDate, AdjustmentType, QuantityAdjusted, Reason, SupplierID) VALUES
(1, 1, '2023-01-15', 'Purchase', 150, 'Initial stock purchase', 1),
(2, 1, '2023-02-20', 'Sale', -30, 'Monthly sales', NULL),
(3, 1, '2023-03-22', 'Sale', -50, 'Promotional event', NULL),
(4, 1, '2023-07-20', 'Purchase', 50, 'Restock due to high demand', 2),
(5, 2, '2023-02-10', 'Purchase', 50, 'Restock', 1),
(6, 2, '2023-03-15', 'Sale', -20, 'Online sales', NULL),
(7, 2, '2023-08-25', 'Sale', -10, 'Summer sale', NULL),
(8, 3, '2023-01-05', 'Purchase', 300, 'Bulk purchase', 3),
(9, 3, '2023-04-18', 'Sale', -100, 'Corporate sales', NULL),
(10, 3, '2023-09-30', 'Sale', -50, 'End of quarter sale', NULL),
(11, 4, '2023-02-25', 'Purchase', 25, 'Restock', 2),
(12, 4, '2023-05-30', 'Sale', -10, 'Retail sales', NULL),
(13, 4, '2023-10-10', 'Purchase', 10, 'Minor restock', 1),
(14, 5, '2023-01-01', 'Obsolescence', -10, 'Write-off obsolete stock', NULL),
(15, 5, '2023-06-15', 'Sale', -5, 'Clearance sale', NULL);
