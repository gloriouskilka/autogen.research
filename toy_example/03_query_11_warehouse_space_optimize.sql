-- Optimize Warehouse Space
-- Analyze inventory distribution across warehouses and identify space utilization.

SELECT
    i.WarehouseLocation,
    COUNT(i.BookID) AS NumberOfBooks,
    SUM(i.QuantityOnHand) AS TotalUnits,
    AVG(b.Price) AS AverageBookPrice,
    SUM(i.QuantityOnHand * b.Price) AS TotalInventoryValue
FROM
    INVENTORY i
    JOIN BOOKS b ON i.BookID = b.BookID
GROUP BY
    i.WarehouseLocation
ORDER BY
    TotalInventoryValue DESC;

 /*
 Explanation:
 - NumberOfBooks: Number of different book titles stored in each warehouse location.
 - TotalUnits: Total units of books in each warehouse.
 - AverageBookPrice: Average selling price of books in each warehouse.
 - TotalInventoryValue: Total value of inventory stored in each warehouse.
 - Helps in identifying which warehouses are overstocked or underutilized.

Actionable Steps:
Redistribute Inventory: Move excess stock from overstocked warehouses to locations with higher demand.
Consolidate Warehouses: Reduce the number of warehouses if some are consistently underutilized, saving on rental and operational costs.
Improve Layouts: Optimize shelf arrangements based on product popularity and turnover rates to enhance accessibility and picking efficiency.
 */
