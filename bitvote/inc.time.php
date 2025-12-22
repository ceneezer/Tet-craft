<?php
function Convert_To_Asgardian_Time(string $timestamp) {
   $A01=1451606399;//2016-01-01
   $sec=1;
   $min=60*$sec;
   $hour=60*$min;
   $day=24*$hour;
   $month=28*$day;
   $year=13*$month+1;
   $timestamp=strtotime($timestamp);
   $timestamp-=$A01;
   $y=intval($timestamp/$year);
   $timestamp-=$year*$y;
   $m=intval($timestamp/$month);
   $timestamp-=$month*$m;
   $d=intval($timestamp/$day);
   $timestamp-=$day*$d;
   $h=intval($timestamp/$hour);
   $timestamp-=$hour*$h;
   $i=intval($timestamp/$min);
   $timestamp-=$min*$i;
   $s=intval($timestamp/$sec);
   $timestamp-=$sec*$s;
   $month=[
      "January",
      "Febuary",
      "March",
      "April",
      "May",
      "June",
      "Asgard",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December"];
   $y=sprintf('%04d', $y);
   $month=$month[$m];
   $m=sprintf('%02d', $m);
   $d=sprintf('%02d', $d);
   $i=sprintf('%02d', $i);
   $s=sprintf('%02d', $s);
   $r=["A$y-$m-$d $h:$i:$s","$month $d, A$y"]; 
   return $r;
}
?>