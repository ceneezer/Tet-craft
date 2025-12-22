USE voteprism;
SET NAMES utf8mb4;

#DROP TABLE IF EXISTS sqlate;
CREATE TABLE sqlate (
    id SERIAL PRIMARY KEY,
    encaption VARCHAR(255) NOT NULL,
    frcaption VARCHAR(255)
);

#DROP TABLE IF EXISTS Psessions;
#CREATE TABLE Psessions (
#   id SERIAL PRIMARY KEY,
#   caption CHAR(4), 
#   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#   UNIQUE (caption) 
#);
#INSERT INTO Psessions (caption) values ('38-1'), ('39-1'), ('39-2'), ('40-2'), ('40-3'), ('41-1'), ('41-2'), ('42-1'), ('43-1'), ('43-2'), ('44-1');
#UPDATE Psessions SET stamp=null WHERE caption IN ('38-1', '39-1', '39-2', '40-2', '40-3', '41-1', '41-2');
#
#DROP TABLE IF EXISTS regions;
#CREATE TABLE regions (
#   id SERIAL PRIMARY KEY,
#   caption CHAR(255), 
#   frcaption CHAR(255)
#);
#INSERT INTO regions (caption, frcaption) SELECT DISTINCT region, fregion FROM users;
#UPDATE users AS u INNER JOIN regions AS r ON r.caption=u.region SET u.ID_region=r.id;
#
#DROP TABLE IF EXISTS parties;
#CREATE TABLE parties (
#   id SERIAL PRIMARY KEY,
#   caption CHAR(255), 
#   frcaption CHAR(255)
#);
#INSERT INTO parties (caption, frcaption) SELECT DISTINCT party, fparty FROM users;
#UPDATE users AS u INNER JOIN parties AS p ON p.caption=u.party SET u.ID_party=p.id;
#
#DROP TABLE IF EXISTS users;
#CREATE TABLE users (
#   id SERIAL PRIMARY KEY,
#   ssid BIGINT UNSIGNED,
#   mpid VARCHAR(40),
#   ID_party BIGINT UNSIGNED,
#   fname VARCHAR(50),
#   lname VARCHAR(50),
#   phone VARCHAR(25),
#   email VARCHAR(255) UNIQUE,
#   ID_region BIGINT UNSIGNED,
#   link VARCHAR(255),
#   passhash CHAR(128),
#   laston TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#);
#ALTER TABLE users ADD UNIQUE mky(ssid, mpid);
#ALTER TABLE users ADD FOREIGN KEY (ID_party) REFERENCES parties(id) ON DELETE RESTRICT;
#ALTER TABLE users ADD FOREIGN KEY (ID_region) REFERENCES regions(id) ON DELETE RESTRICT;
#
#DROP TABLE IF EXISTS logs;
#CREATE TABLE logs (
#   id SERIAL PRIMARY KEY,
#   ID_user BIGINT UNSIGNED NOT NULL,
#   IP VARCHAR(25) NOT NULL,
#   caption TEXT,
#   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
#);



DROP TABLE IF EXISTS Psessions;
CREATE TABLE Psessions (
   id SERIAL PRIMARY KEY,
   caption CHAR(4), 
   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (caption) 
);
INSERT INTO Psessions (caption) values ('38-1'), ('39-1'), ('39-2'), ('40-2'), ('40-3'), ('41-1'), ('41-2'), ('42-1'), ('43-1'), ('43-2'), ('44-1');
UPDATE Psessions SET stamp=null WHERE caption IN ('38-1', '39-1', '39-2', '40-2', '40-3', '41-1', '41-2');

DROP TABLE IF EXISTS regions;
CREATE TABLE regions (
   id SERIAL PRIMARY KEY,
   caption CHAR(255) UNIQUE, 
   frcaption CHAR(255)
);
INSERT INTO regions (caption, frcaption) SELECT DISTINCT region, fregion FROM users;

DROP TABLE IF EXISTS parties;
CREATE TABLE parties (
   id SERIAL PRIMARY KEY,
   caption CHAR(255) UNIQUE, 
   frcaption CHAR(255)
);
INSERT INTO parties (caption, frcaption) SELECT DISTINCT party, fparty FROM users;
UPDATE users AS u INNER JOIN parties AS p ON p.caption=u.party SET u.ID_party=p.id;

DROP TABLE IF EXISTS users;
CREATE TABLE users (
   id SERIAL PRIMARY KEY,
   ssid BIGINT UNSIGNED,
   mpid VARCHAR(40),
   ID_party BIGINT UNSIGNED,
   fname VARCHAR(50),
   lname VARCHAR(50),
   phone VARCHAR(25),
   email VARCHAR(255) UNIQUE,
   ID_region BIGINT UNSIGNED,
   link VARCHAR(255),
   passhash CHAR(128),
   laston TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
ALTER TABLE users ADD UNIQUE mky(ssid, mpid);
ALTER TABLE users ADD FOREIGN KEY (ID_party) REFERENCES parties(id) ON DELETE RESTRICT;
ALTER TABLE users ADD FOREIGN KEY (ID_region) REFERENCES regions(id) ON DELETE RESTRICT;
UPDATE users AS u INNER JOIN regions AS r ON r.caption=u.region SET u.ID_region=r.id;

DROP TABLE IF EXISTS logs;
CREATE TABLE logs (
   id SERIAL PRIMARY KEY,
   ID_user BIGINT UNSIGNED NOT NULL,
   IP VARCHAR(25) NOT NULL,
   caption TEXT,
   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS issues;
CREATE TABLE issues(
   id SERIAL PRIMARY KEY,
   deadline DATETIME,
   ssid BIGINT UNSIGNED,
   mpid VARCHAR(255),
   title VARCHAR(255),
   frtitle VARCHAR(255),
   journal VARCHAR(255), 
   leginfo VARCHAR(255), 
   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE issues ADD UNIQUE mky(ssid, mpid);

DROP TABLE IF EXISTS votes;
CREATE TABLE votes(
   id SERIAL PRIMARY KEY,
   Vvalue BIGINT NOT NULL,
   caption VARCHAR(10) NOT NULL UNIQUE,
   frcaption VARCHAR(10) NOT NULL
);
INSERT INTO votes (Vvalue, caption, frcaption) VALUES (0, 'abs', 'abstension');
INSERT INTO votes (Vvalue, caption, frcaption) VALUES (1, 'yea', 'pour');
INSERT INTO votes (Vvalue, caption, frcaption) VALUES (-1, 'nay', 'contre');

DROP TABLE IF EXISTS issue_votes;
CREATE TABLE issue_votes(
   id SERIAL PRIMARY KEY,
   ID_issue BIGINT UNSIGNED,
   ID_user BIGINT UNSIGNED,
   ID_vote BIGINT UNSIGNED,
   stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
ALTER TABLE issue_votes ADD UNIQUE mky(ID_issue, ID_user);
ALTER TABLE issue_votes ADD FOREIGN KEY (ID_vote) REFERENCES votes(id) ON DELETE RESTRICT;
ALTER TABLE issue_votes ADD FOREIGN KEY (ID_issue) REFERENCES issues(id) ON DELETE RESTRICT;
ALTER TABLE issue_votes ADD FOREIGN KEY (ID_user) REFERENCES users(id) ON DELETE RESTRICT;

DROP TABLE IF EXISTS tags;
CREATE TABLE tags (
   id SERIAL PRIMARY KEY,
   caption VARCHAR(25) UNIQUE
);

DROP TABLE IF EXISTS issue_tags;
CREATE TABLE issue_tags (
   ID_issue BIGINT UNSIGNED,
   ID_user BIGINT UNSIGNED,
   ID_tag BIGINT UNSIGNED
);
ALTER TABLE issue_tags ADD PRIMARY KEY id(ID_issue, ID_user, ID_tag);
ALTER TABLE issue_tags ADD FOREIGN KEY (ID_tag) REFERENCES tags(id) ON DELETE RESTRICT;
ALTER TABLE issue_tags ADD FOREIGN KEY (ID_issue) REFERENCES issues(id) ON DELETE RESTRICT;
ALTER TABLE issue_tags ADD FOREIGN KEY (ID_user) REFERENCES users(id) ON DELETE RESTRICT;

DROP TABLE IF EXISTS user_tags;
CREATE TABLE user_tags(
   ID_user BIGINT UNSIGNED,
   ID_tag BIGINT UNSIGNED
);
ALTER TABLE user_tags ADD PRIMARY KEY id(ID_user, ID_tag);
ALTER TABLE user_tags ADD FOREIGN KEY (ID_tag) REFERENCES tags(id) ON DELETE RESTRICT;
ALTER TABLE user_tags ADD FOREIGN KEY (ID_user) REFERENCES users(id) ON DELETE RESTRICT;

DROP TABLE IF EXISTS scores;
CREATE TABLE scores ( 
   ID_issue BIGINT UNSIGNED,
   ID_user BIGINT UNSIGNED,
   ID_vote BIGINT UNSIGNED,
   bID_vote BIGINT UNSIGNED,
   bID_user BIGINT UNSIGNED
);
ALTER TABLE scores ADD PRIMARY KEY id(ID_user, ID_issue, bID_user);

CREATE OR REPLACE VIEW joined_votes AS
   SELECT i.id AS _ID, i.ID_issue, i.ID_user, i.ID_vote, v.caption as vote, 
      r.caption as region, p.caption as party, u.* 
      FROM issue_votes AS i 
         LEFT JOIN votes AS v ON v.id=i.ID_vote 
         LEFT JOIN users AS u ON u.id=i.ID_user 
         LEFT JOIN regions AS r ON r.id=u.ID_region 
         LEFT JOIN parties AS p ON p.id=u.ID_party;

DROP PROCEDURE IF EXISTS doScores;
CREATE PROCEDURE doScores() NOT DETERMINISTIC CONTAINS SQL SQL SECURITY DEFINER 
INSERT IGNORE INTO scores 
   (bID_user, bID_vote, ID_issue, ID_user, ID_vote)
   SELECT b.ID_user AS bID_user, b.ID_vote as bID_vote, a.ID_issue, a.ID_user, a.ID_vote
      FROM issue_votes AS a
      RIGHT JOIN issue_votes AS b ON b.ID_issue=a.ID_issue AND NOT b.ID_User=a.ID_User;

#SELECT sum(NULLIF(ID_vote, bID_vote)/ID_vote)/count(bID_vote) AS score, ID_user
#      FROM scores
#   GROUP BY ID_user;
