<?php
if (isset($_POST['phone']) && isset($_POST['link']) 
   && isset($_POST['region']) && isset($_POST['party'])
   && isset($_POST['first']) && isset($_POST['family'])) {
   $first=SQLIFY($_POST['first']);
   if (strlen($first)>50)
      $first=substr($first,0,50);
   $family=SQLIFY($_POST['family']);
   if (strlen($family)>50)
      $family=substr($family,0,50);
   $link=SQLIFY($_POST['link']);
   if (strlen($link)>255)
      $link=substr($link,0,255);
   if ($link=='' || !isset($_POST['public']))
      $link='null';
   else
      $link='"'.$link.'"';
   $sql='UPDATE users SET link='.$link.', fname="'
      .$first.'", lname="'.$family.'"';
   $phone=SQLIFY($_POST['phone']);
   if (strlen($phone)<25)
      $sql.=', phone="'.$phone.'"';
   $region=intval($_POST['region']);
   if ($region>0)
      $sql.=', ID_region="'.$region.'"';
   else  
      $sql.=', ID_region=null';
   $party=intval($_POST['party']);
   if ($party>0)
      $sql.=', ID_party="'.$party.'"';
   else  
      $sql.=', ID_party=null';
   $sql.=' WHERE id="'.$_SESSION['UID'].'"';
   SQL2ARRAY($sql);
}
if (isset($_SESSION['UID'])) {
   $sql='SELECT *, (link IS NULL) AS isnull FROM users WHERE id='.$_SESSION['UID'];
   $sql=SQL2ARRAY($sql)[0];
?>
   <main>
      <form class="flex me" method="post">
         <span>
<input type="submit" value="<?=SQLATE('update', $LANG, true)?>" onmousedown="document.body.style.cursor='wait';this.classList.add('wait');" />
<script>
function nableLink(chk) {
   var lnk=document.getElementById('link');
   if (chk) {
      lnk.classList.remove("hidden");
   } else {
      lnk.classList.add("hidden");
      lnk.value='';
   }
}
</script>
<label><input name="public" onchange="nableLink(this.checked);" type="checkbox"<?=($sql['isnull']=='1')?'':' checked="checked"'?>/>Public</label>
         </span>
<?php
$drp='SELECT id, ';
if ($LANG=='fr')
   $drp.='fr';
$drp.='caption AS caption FROM parties ORDER BY ';
if ($LANG=='fr')
   $drp.='fr';
$drp.='caption';
$drp=SQL2ARRAY($drp);
echo '<select name="party"><option value="0">('.SQLATE('select your party', $LANG, true).')</option>'."\n";
foreach ($drp as $d) {
   echo '<option value="'.$d['id'];
   if ($d['id']==$sql['ID_party'])
      echo '" selected="selected"';
   echo '">'.$d['caption'].'</option>'."\n";
}
echo '</select>'."\n";
$drp='SELECT id, ';
//if ($LANG=='fr')
//   $drp.='fr';
$drp.='caption AS caption FROM regions ORDER BY caption';
$drp=SQL2ARRAY($drp);
echo '<select name="region"><option value="0">('.SQLATE('select your region', $LANG, true).')</option>'."\n";
foreach ($drp as $d) {
   echo '<option value="'.$d['id'];
   if ($d['id']==$sql['ID_region'])
      echo '" selected="selected"';
   echo '">'.$d['caption'].'</option>'."\n";
}
echo '</select>'."\n";
?>
<input size="10" name="phone" type="phone" value="<?=$sql['phone']?>" placeholder="XXX-YYY-ZZZZ xWWW" title="XXX-YYY-ZZZZ xWWW" />
<span><?=$sql['email']?></span>
<input size="10" name="first" value="<?=$sql['fname']?>" placeholder="<?=SQLATE('given name', $LANG, true)?>" title="<?=SQLATE('given name', $LANG, true)?>"/>
<input size="10" name="family" value="<?=$sql['lname']?>" placeholder="<?=SQLATE('family name', $LANG, true)?>" title="<?=SQLATE('family name', $LANG, true)?>"/>
<input id="link" value="<?=$sql['link']?>" class="<?=($sql['isnull']=='1')?'hidden':''?>" name="link" placeholder="<?=SQLATE('candidacy page', $LANG, true)?>" title="<?=SQLATE('candidacy page', $LANG, true)?>" />
      </form>
      <section>
<?php
//if search
   $sql='SELECT u.*, p.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as pparty, r.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as rregion, COUNT(vf.Vvalue)-COUNT(va.Vvalue) as score FROM issue_votes AS i LEFT JOIN issue_votes AS b ON b.ID_issue=i.ID_issue AND b.ID_user="'
      .SQLIFY($_SESSION['UID'])
      .'" LEFT JOIN votes AS vf ON vf.id=i.ID_vote AND i.ID_vote=b.ID_vote LEFT JOIN votes AS va ON va.id=i.ID_vote AND NOT i.ID_vote=b.ID_vote LEFT JOIN users AS u ON u.id=i.ID_user AND NOT u.link IS NULL AND NOT u.link="" LEFT JOIN parties AS p ON p.id=u.ID_party LEFT JOIN regions AS r ON r.id=u.ID_region WHERE NOT u.link IS NULL';
   if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
      $search=str_replace('%', '', $search);
      $search=SQLIFY($_REQUEST['search']);
      $sql.=' AND (';
      $sql.='p.caption LIKE "%'.$search.'%"';
      $sql.=' OR p.frcaption LIKE "%'.$search.'%"';
      $sql.=' OR r.caption LIKE "%'.$search.'%"';
      $sql.=' OR r.frcaption LIKE "%'.$search.'%"';
      $sql.=' OR u.fname LIKE "%'.$search.'%"';
      $sql.=' OR u.lname LIKE "%'.$search.'%"';
      $sql.=' OR u.link LIKE "%'.$search.'%"';
      $sql.=' OR u.phone LIKE "%'.$search.'%"';
      $sql.=' OR u.email LIKE "%'.$search.'%"';
      $sql.=' OR u.ssid LIKE "%'.$search.'%"';
      $sql.=' OR u.mpid LIKE "%'.$search.'%"';
      $sql.=')';
   }
   $sql.=' GROUP BY u.id ORDER BY score DESC';
   $sql=SQL2ARRAY($sql);
   if (count($sql)>0) {
?>
         <table id="tblVotes">
            <tr>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 0)"><?=SQLATE('politician', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 1)"><?=SQLATE('score', $LANG, true)?></th>
            </tr>
<?php foreach ($sql as $s) { ?>
            <tr>
               <td style="text-align:left;" >
                  <span class="flex">
<span class="hidden"><?=$s['lname']?> <?=$s['fname']?></span>
<span><?=$s['phone']?></span>
<span><?=$s['email']?></span>
<a href="<?=$s['link']?>" target="_blank"><?=$s['fname']?> <?=$s['lname']?></a>
<span><?=$s['pparty']?>
<?php if ($s['ssid']>0) {?>
 <a href="https://sencanada.ca/<?=($LANG=='fr')?'fr/dans-la-chambre':'en/in-the-chamber'?>/votes/senator/<?=$s['ssid']?>/" target="_blank"><?=SQLATE('chamber', $LANG, false)?></a>
<?php } ?>
</span>
               </td>
               <td>
<a href='?page=profile&id=<?=$s['id']?>'><?=$s['score']?></a></td>
            </tr>
<?php } ?>
         </table>
      </section>
   </main>
<?php }} ?>