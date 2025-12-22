<?php
require_once 'inc.db.php';
require_once '/srv/www/inc.SQL2ARRAY.php';
require_once '/srv/www/inc.MY_LOG.php';

function gUIDv4($data=null, $table=false, $field=false) {
   // Generate 16 bytes (128 bits) of random data or use the data passed into the function.
   $data = $data ?? random_bytes(16);
   assert(strlen($data) == 16);   
   // Set version to 0100   
   $data[6] = chr(ord($data[6]) & 0x0f | 0x40);
   // Set bits 6-7 to 10
   $data[8] = chr(ord($data[8]) & 0x3f | 0x80);
   $ret=vsprintf('%s%s-%s-%s-%s-%s%s%s', str_split(bin2hex($data), 4));
   if ($table && $field) {
      //ensure not in table, field
      $sql=SQL2ARRAY('SELECT id FROM '.$table.' WHERE '.$field.'="'.SQLIFY($ret).'"');
      if (count($sql)>0)
         return gUIDv4(null, $table, $field);
   }
   // Output the 36 character UUID.
   return $ret;
}

function SQLIFY($text, $wildcard=false) {
   global $con;
   $text=str_replace('/', '&sol;', $text);
   $text=str_replace("\\", '&bsol;', $text);
   $text=str_replace('<', '&lt;', $text);
   $text=str_replace('>', '&gt;', $text);
   $text=str_replace('{', '&#123;', $text);
   $text=str_replace('}', '&#125;', $text);
   $text=str_replace("'", '&apos;', $text);
   $text=str_replace('"', '&quot;', $text);
   if (!$wildcard) {
      $text=str_replace('_', '&lowbar;', $text);
      $text=str_replace('%', '&percnt;', $text);
   }
   $text=str_replace('�', 'é', $text);
   $text=trim($text);
   $text=mysqli_real_escape_string($con, $text);
   return $text;
} ?>
