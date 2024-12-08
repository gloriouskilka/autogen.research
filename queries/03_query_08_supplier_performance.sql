-- Evaluate Supplier Performance
-- Assess based on number of orders, average delivery time, and reliability score.

SELECT
    s.SupplierID,
    s.SupplierName,
    COUNT(a.AdjustmentID) AS TotalOrders,
    AVG(s.DeliveryTimeDays) AS AverageDeliveryTime,
    s.ReliabilityScore
FROM
    SUPPLIERS s
    JOIN INVENTORY_ADJUSTMENTS a ON s.SupplierID = a.SupplierID
WHERE
    a.AdjustmentType = 'Purchase'
GROUP BY
    s.SupplierID,
    s.SupplierName,
    s.ReliabilityScore
ORDER BY
    s.ReliabilityScore DESC,
    TotalOrders DESC;

 /*
 Explanation:
 - TotalOrders: Number of purchase orders from each supplier.
 - AverageDeliveryTime: Average number of days taken to deliver orders.
 - ReliabilityScore: A predefined score indicating overall supplier reliability.
 - Helps in identifying top-performing suppliers and those needing improvement.

Actionable Steps:
Strengthen Relationships with Top Suppliers: Prioritize and strengthen partnerships with reliable suppliers.
Address Issues with Underperforming Suppliers: Communicate performance concerns and consider alternative suppliers if necessary.
Negotiate Better Terms: Use performance data to negotiate better pricing, terms, or services with suppliers.
 */
