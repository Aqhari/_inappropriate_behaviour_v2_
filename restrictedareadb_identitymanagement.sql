-- MySQL dump 10.13  Distrib 8.0.40, for Win64 (x86_64)
--
-- Host: localhost    Database: restrictedareadb
-- ------------------------------------------------------
-- Server version	9.1.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `identitymanagement`
--

DROP TABLE IF EXISTS `identitymanagement`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `identitymanagement` (
  `ID` int NOT NULL AUTO_INCREMENT,
  `PersonName` varchar(100) NOT NULL,
  `Certificate1` date NOT NULL,
  `Certificate2` tinyint(1) NOT NULL,
  `Certificate3` date NOT NULL,
  `Certificate4` tinyint(1) NOT NULL,
  PRIMARY KEY (`ID`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `identitymanagement`
--

LOCK TABLES `identitymanagement` WRITE;
/*!40000 ALTER TABLE `identitymanagement` DISABLE KEYS */;
INSERT INTO `identitymanagement` VALUES (1,'Abdul Kadir','2026-05-15',1,'2025-11-20',1),(2,'Aqil Farhan','2024-08-10',0,'2023-12-01',0),(3,'Arif Aiman','2027-03-01',1,'2026-07-25',1),(4,'Maryam Sufia','2025-01-15',1,'2024-12-07',1),(5,'Mohd Anuar Rosly','2025-11-23',0,'2025-12-11',0),(6,'Muhammad Aqhari Nasrin','2025-02-24',1,'2024-02-17',0),(7,'Nur Atiqah Binti Pauzan','2020-02-03',1,'2020-07-20',0),(8,'Shah Al Haffiz','2025-12-23',1,'2025-01-25',1),(9,'Muhammad Afiq Bin Nooralhadi','2027-07-18',1,'2027-05-12',1);
/*!40000 ALTER TABLE `identitymanagement` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-06-19 11:24:43
