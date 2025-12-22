<?php
require_once '/srv/www/inc.db.php';
require_once '/srv/www/inc.SQLIFY.php';
require_once '/srv/www/inc.SQL2ARRAY.php';

function SQLTIME($text) {
   if ($text=='') return "null"; 
   return 
      date("'Y-m-d H:i:s'"
         , strtotime($text));
}

function MY_LOG ($text) {
	$text=SQLIFY($text);  //doublechecks entry is safe
   if (strlen($text)>65000)
      $text=substr($text,0,65000);
	$sql="INSERT logs SET IP ='".$_SERVER['REMOTE_ADDR']
		."', caption='$text' , ID_user=0";
	if ( isset( $_SESSION['UID'] ) ) 
		$sql .= $_SESSION['UID'];
	SQL2ARRAY($sql); // sends sql command
}

//SQL2ARRAY("SET time_zone = '-4:00'");
/*
DROP TABLE IF EXISTS logs;
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    ID_user BIGINT UNSIGNED NOT NULL,
    IP VARCHAR(25) NOT NULL,
    caption TEXT,
    stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
*/
?>
