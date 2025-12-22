<?php 
require_once 'inc.db.php';
require_once '/srv/www/inc.SQLIFY.php';
require_once '/srv/www/inc.MY_LOG.php';

function SQL2ARRAY($sql) {
	global $con;                                      //expects a $con already from inc.db.php
	$return=array();								  //creates am empty array to return
//	echo $sql;
	$result=mysqli_query($con, $sql);				  //execute query
//    var_dump($result);
	if ($result===false){							  //insert, ddelete, etc failed with error
MY_LOG($sql." - ".mysqli_error($con));
		die(mysqli_error($con));}						  //may prefer to output/logerror/notdie
	if ($result===true){return $return;}
	if (mysqli_num_rows($result)>0){
		while ($row=mysqli_fetch_array($result)){
			$return[]=$row;}}
	return $return;	
} ?>