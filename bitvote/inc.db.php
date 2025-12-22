<?php
date_default_timezone_set('America/New_York');
$con=mysqli_connect('localhost',$DBUN,$DBPW,$DBDB);
if ($con -> connect_errno) {
  echo "Failed to connect to MySQL: " . $con -> connect_error;
  exit();
}
mysqli_set_charset($con,"utf8mb4");
$salt=['ceneezer','opWorldPeace']; //primary author
error_reporting(E_ALL & ~E_WARNING & ~E_NOTICE & ~E_USER_DEPRECATED);
?>
