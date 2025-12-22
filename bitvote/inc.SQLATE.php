<?php
require_once '/srv/www/inc.db.php';
require_once '/srv/www/inc.SQLIFY.php';
require_once '/srv/www/inc.SQL2ARRAY.php';

function SQLATE ($text, $lang, $capitalize) {
	$text=SQLIFY($text);  //doublechecks entry is safe
   $lang=SQLIFY($lang);
	$sql='SELECT '.$lang.'caption FROM sqlate WHERE encaption="'.$text.'"';
	$sql=SQL2ARRAY($sql); // sends sql command
   if (count($sql)>0) {
      if ($capitalize)
         return ucwords($sql[0][0]);
      else
         return $sql[0][0];
   }
   return false;
}

/*
DROP TABLE IF EXISTS sqlate;
CREATE TABLE sqlate (
    id SERIAL PRIMARY KEY,
    encaption VARCHAR(255) NOT NULL,
    frcaption VARCHAR(255)
);
*/
?>